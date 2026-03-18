import re
import json
import yaml
import logging
import argparse

from pathlib import Path
from jinja2 import Template
from typing import Any
from concurrent.futures import ThreadPoolExecutor,as_completed
from langchain_openai import ChatOpenAI

from models.llm import get_llm,invoke_with_retry
from memory.grpo_memory import Memory
from models.judge_llm import judge
from utils.process_benchmark import process_parquet_benchmark
from utils.set_logger import setup_logger,format_log_without_judge,set_filehandler

DEFAULT_MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507"
DEFAULT_BENCHMARK = "wmdp"
MAX_NUM = 20

def test_single_question(llm,data,memories,test_sys,test_user_template):
    test_user = test_user_template.render(
        question = data["question"],
        choices = data["choices"],
        memory = memories,
    )
    try:
        response = invoke_with_retry(
            llm,
            {"role":"user","content":test_user},
            message_history = [{"role":"system","content":test_sys}]
        )
    except:
        response = "The agent failed to get an answer due to connection error."
    return response

def multitest_single_question(llm,data,memories,test_sys,test_user_template,times,max_workers = 10):
    responses = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {
            executor.submit(test_single_question,llm,data,memories,test_sys,test_user_template):id
            for id in range(times)
        }
        for future in as_completed(futures):
            responses.append(future.result())
    
    #exclude judge result
    corrects = []
    for response in responses: 
        answer_pattern = r"</scrachpad>.*?<answer>(.*?)</answer>"
        correct = 2
        match = re.search(answer_pattern, response.strip(), re.DOTALL)
        if match :
            try:
                answer = int(match.group(1).strip())
                if answer == data["answer"]:
                    correct = 1
                else:
                    correct = 0
            except:
                correct = 2
        corrects.append(correct)
    
    return (responses,corrects)

def test_batch_questions(llm,test_data,start_id,end_id,memories,test_sys,test_user_template,times,logger,max_workers = 10):
    results = {
        "success":[],
        "failure":{"answer":[],"parse":[]}
    }
    model_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(multitest_single_question,llm,test_data[id],memories,test_sys,test_user_template,times,max_workers):id
            for id in range(start_id,end_id + 1)
        }
        
        for future in as_completed(futures):
            responses,corrects = future.result()
            id = futures[future]
            logger_info = format_log_without_judge(test_data[id], responses, corrects)
            logger.info(logger_info)
            id = futures[future]
            model_results[id] = {
                "question":test_data[id]["question"],
                "choices":test_data[id]["choices"],
                "responses":responses
            }
            for response,correct in zip(responses,corrects):
                status = {
                    "question":test_data[id]["question"],
                    "choices":test_data[id]["choices"],
                    "response":response
                }
                if correct == 1:
                    results["success"].append(status)
                elif correct == 0 :
                    results["failure"]["answer"].append(status)
                else:
                    results["failure"]["parse"].append(status)
    return results,model_results

def extract_memory(llm,memories,results,memory_prompts):
    user = Template(memory_prompts["user_template"]).render(results = results,memories = memories)

    try :
        memory = invoke_with_retry(
            llm,
            {"role":"user","content":user},
            [{"role":"system","content":memory_prompts["sys_template"]}]
        )
    except:
        memory = "No success memory." 
    
    return memory

def main():
    parser = argparse.ArgumentParser(description="setup the parameter for test")
    parser.add_argument("--model","-m",default = DEFAULT_MODEL_NAME,help="the model you want to test")
    parser.add_argument("--benchmark","-b",default=DEFAULT_BENCHMARK,help="the benchmark you want to test")
    parser.add_argument("--times","-t",type=int,default=3,help = "the number of times a question will be tested")
    parser.add_argument("--max_workers","-n",type=int,default=10,help = "max workers for ThreadPoolExecutors")
    parser.add_argument("--batch",type=int,default=8,help = "max workers for ThreadPoolExecutors")
    args = parser.parse_args()
    args = parser.parse_args()

    model = args.model
    benchmark = args.benchmark
    times = args.times
    test_prompt_path = "configs/test_template.yml"
    memory_prompt_path = "configs/reasoning_bank_template.yml"
    grpo_prompt_path = "configs/grpo_template.yml"
    max_workers = args.max_workers
    batch = args.batch
    max_tokens = 8192
    temperature = 0.7

    logger: logging.Logger
    logger,logger_name,log_dir = setup_logger("test_without_judge",model_name = model,benchmark_name = benchmark)
    set_filehandler(logger,log_dir,"setup_info")
    logger.info("="*20 + " 🛠️  System Initialization " + "="*20)
    try:
        llm = get_llm(model_name = model,temperature = temperature,max_tokens = max_tokens)
        memory_llm = get_llm(model_name = model,temperature = temperature,max_tokens = None)
        logger.info(f"🤖 Model loaded: [ {model} ] (Temp: 0.7) (Max_tokens = {max_tokens})")
        #embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        grpo_memory = Memory()
        logger.info("🧠 Memory module: Initialized successfully")
        test_data = process_parquet_benchmark("dataset/" + benchmark)
        sub_task_count = len(test_data)
        total_questions = sum(len(v) for v in test_data.values())
        logger.info(f"📊 Dataset loaded: '{benchmark}' | Sub-tasks: {sub_task_count} | Total Questions: {total_questions}")
        #answer = invoke_with_retry(llm,{"role":"user","content":"introduce yourself"})
        #print(answer)
        with open(test_prompt_path,"r",encoding = "utf-8") as f:
            prompts = yaml.safe_load(f)
        test_sys = prompts["test_system_template"]
        test_user_template = Template(prompts["test_user_template"] +  prompts["user_template_with_memory"])
       
        with open(memory_prompt_path,"r",encoding = "utf-8") as f:
            memory_prompts = yaml.safe_load(f)
        with open (grpo_prompt_path,"r",encoding = "utf-8") as f:
            grpo_prompts = yaml.safe_load(f)
        grpo_sys = grpo_prompts["grpo_sys_template"]
        grpo_user_template = Template(grpo_prompts["grpo_user_template"])

        # 1. 准备要打印的 Prompt 字典，方便统一记录
        all_prompts = {
            "test_sys": test_sys,
            "test_user_template_raw": prompts["test_user_template"] +  prompts["user_template_with_memory"],
            "memory_sys": memory_prompts["sys_template"],
            "memory_user_template_raw":memory_prompts["user_template"],
            "grpo_sys":grpo_sys,
            "grpo_user_template_raw": grpo_prompts["grpo_user_template"],
        }

        # 2. 详细记录到 Logger
        logger.info("🎨 Prompt templates: All Jinja2 templates compiled")
        logger.info("--- Detailed Prompt Templates Content ---")

        for name, content in all_prompts.items():
            # 使用分隔符包裹内容，防止多行文本混淆
            logger.info(f"\n>>> TEMPLATE NAME: {name} <<<\n{content}\n" + "-"*30)

        logger.info("--- End of Prompt Templates ---")
        logger.info("-" * 50)
    except FileNotFoundError as e:
        logger.critical(f"🛑 Critical Configuration Error: {e}")
        return # 优雅退出
    except yaml.YAMLError as e:
        logger.critical(f"🛑 YAML Parsing Error: {e}")
        return
    except Exception as e:
        logger.warning(f"🛑 Setup fail:{e}")
    

    correct_num, false_num, parse_error_num = 0,0,0
    batch_id = 0
    total_processed_questions = 0
    try :
        for sub_benchmark,subtest_data in test_data.items():  
            start_id = 0 
            if not sub_benchmark.startswith("wmdp-cyber"):continue
            while start_id < len(subtest_data):
                set_filehandler(logger,log_dir,f"batch_{batch_id}")
                logger.info("="*30 + " 🚀 Starting New Batch " + "="*30)
                logger.info(f"📍 Sub-benchmark: {sub_benchmark} | Range: [{start_id} - {start_id + batch - 1}]")
                new_results,model_results = test_batch_questions(llm,subtest_data,start_id,min(start_id + batch - 1,len(subtest_data) - 1),grpo_memory.format_memories(),test_sys,test_user_template,times,logger,max_workers)
                memory = extract_memory(memory_llm,grpo_memory.format_memories(),model_results,memory_prompts)
                grpo_memory.process_new_memory(memory)
               
                logger.debug(f"📝 Raw Insight:\n {memory}...")      
                logger.info(f"📝 Raw Memory Module Output:\n {grpo_memory.format_memories()}")#memories
                grpo_user_prompt = grpo_user_template.render(memories = grpo_memory.format_memories())

                try :
                    response_opt_memory = invoke_with_retry(
                        memory_llm,
                        {
                            "role":"user",
                            "content":grpo_user_prompt,
                        },
                        [{"role":"system","content":grpo_sys}]
                    )
                except:
                    response_opt_memory = "Fail to get memory optimization plan."
                logger.info(f"🤖 Optimizing memory policy via LLM reasoning (GRPO)...\n {response_opt_memory}")
                grpo_memory.process_opt_memory(response_opt_memory)
                logger.info("✨" + "-" * 20 + " Updated Global Memory " + "-" * 20 + "✨")
                logger.info(f"\n{grpo_memory.format_memories()}")
                logger.info("✨" + "-" * 60 + "✨")
                
                correct_num += len(new_results["success"])
                false_num += len(new_results["failure"]["answer"])
                parse_error_num += len(new_results["failure"]["parse"])
                
                start_id += batch
                total_processed_questions += batch
                batch_id += 1
                if start_id >= 100: break
    except KeyboardInterrupt:
        logger.warning("🛑 User interrupted the process. Saving current metadata...")
    except Exception as e:
        logger.error(f"💥 Unexpected system crash: {e}", exc_info=True)
            
    try:
        metadata = {
            "model":model,
            "benchmark":benchmark,
            "times":times,
            "max_workers":max_workers,
            "correct_num":correct_num,
            "false_num":false_num,
            "parse_error_num":parse_error_num
        }

        meta_file =f"{log_dir}/{logger_name}.json"
        with open(meta_file,"w",encoding = "utf-8") as f:
            json.dump(metadata,f,indent=4)
        logger.info(f"✅ Summary saved to {meta_file}")
    except Exception as e:
        print(f"💥 Final Save Failed: {e}")
       

if __name__ == "__main__":
    main()