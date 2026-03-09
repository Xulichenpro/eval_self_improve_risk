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
from sentence_transformers import SentenceTransformer

from models.llm import get_llm,invoke_with_retry
from memory.grpo_memory import Memory
from models.judge_llm import judge
from utils.process_benchmark import process_parquet_benchmark
from utils.set_logger import setup_logger,format_log

DEFAULT_MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507"
DEFAULT_BENCHMARK = "wmdp"
MAX_NUM = 20

def test_single_question(llm,data,memories,test_sys,test_user_template):
    test_user = test_user_template.render(
        question = data["question"],
        choices = data["choices"],
        memory = memories,
    )
    
    response = invoke_with_retry(
        llm,
        {"role":"user","content":test_user},
        message_history = [{"role":"system","content":test_sys}]
    )
    return response

def multitest_single_question(llm,data,memories,test_sys,test_user_template,judge_sys,judge_user_template,times,max_workers = 10):
    responses = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {
            executor.submit(test_single_question,llm,data,memories,test_sys,test_user_template):id
            for id in range(times)
        }
        for future in as_completed(futures):
            responses.append(future.result())
    judge_user_prompt = judge_user_template.render(data = data,times = times,responses = responses,memory = "/n".join(memories))
    responses = judge(llm,responses,judge_sys,judge_user_prompt)
    
    answer_pattern = r"<answer>(.*?)</answer>"
    correct = 2
    match = re.search(answer_pattern, responses[-1], re.S)
    if match :
        try:
            answer = int(match.group(1).strip())
            if answer == data["answer"]:
                correct = 1
            else:
                correct = 0
        except:
            correct = 2

    return (responses,correct)

def test_batch_questions(llm,test_data,start_id,end_id,memories,test_sys,test_user_template,judge_sys,judge_user_template,times,logger,max_workers = 10):
    results = {
        "success":{},
        "failure":{"answer":{},"parse":{}}
    }
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(multitest_single_question,llm,test_data[id],memories,test_sys,test_user_template,judge_sys,judge_user_template,times,max_workers):id
            for id in range(start_id,end_id + 1)
        }
        
        for future in as_completed(futures):
            responses,correct = future.result()
            id = futures[future]
            logger_info = format_log(test_data[id], responses, correct)
            logger.info(logger_info)
            id = futures[future]
            status = {
                    "question":test_data[id]["question"],
                    "choices":test_data[id]["choices"],
                    "responses":responses
                }
            if correct == 1:
                results["success"][id] = status
            elif correct == 0 :
                results["failure"]["answer"][id] = status 
            else:
                results["failure"]["parse"][id] = status 
    return results
#TODO: 防止超过上下文窗口，对log分块
def extract_memory(llm,results,memory_prompts):
    success_user = Template(memory_prompts["success_trajectory"]["user_template"]).render(results = results["success"])
    fail_user = Template(memory_prompts["fail_trajectory"]["user_template"]).render(results = results["failure"]["answer"] | results["failure"]["parse"])
    success_memory = invoke_with_retry(
        llm,
        {"role":"user","content":success_user},
        [{"role":"system","content":memory_prompts["success_trajectory"]["sys_template"]}]
    )
    fail_memory = invoke_with_retry(
        llm,
        {"role":"user","content":fail_user},
        [{"role":"system","content":memory_prompts["fail_trajectory"]["sys_template"]}]
    )
    return [success_memory,fail_memory]

def main():
    parser = argparse.ArgumentParser(description="setup the parameter for test")
    parser.add_argument("--model","-m",default = DEFAULT_MODEL_NAME,help="the model you want to test")
    parser.add_argument("--benchmark","-b",default=DEFAULT_BENCHMARK,help="the benchmark you want to test")
    parser.add_argument("--times","-t",type=int,default=3,help = "the number of times a question will be tested")
    parser.add_argument("--max_workers","-n",type=int,default=10,help = "max workers for ThreadPoolExecutors")
    parser.add_argument("--batch",type=int,default=3,help = "max workers for ThreadPoolExecutors")
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

    logger:logging.Logger = setup_logger("test_with_memory",model_name = model,benchmark_name = benchmark)

    logger.info("="*20 + " 🛠️  System Initialization " + "="*20)
    try:
        llm = get_llm(model_name = model,temperature = 0.2)
        logger.info(f"🤖 Model loaded: [ {model} ] (Temp: 0.2)")
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
        test_user_template = Template(prompts["test_user_template"] + prompts["user_template_with_memory"])
        judge_sys = prompts["judge_sys_template"]
        judge_user_template = Template(prompts["judege_user_template"] + prompts["user_template_with_memory"])
        with open(memory_prompt_path,"r",encoding = "utf-8") as f:
            memory_prompts = yaml.safe_load(f)
        with open (grpo_prompt_path,"r",encoding = "utf-8") as f:
            grpo_prompts = yaml.safe_load(f)
        grpo_prompt_template = Template(grpo_prompts["grpo_user_template"])
        logger.info("🎨 Prompt templates: All Jinja2 templates compiled")
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

    try :
        for sub_benchmark,subtest_data in test_data.items():  
            start_id = 0 
            while start_id < len(subtest_data):
                logger.info("="*30 + " 🚀 Starting New Batch " + "="*30)
                logger.info(f"📍 Sub-benchmark: {sub_benchmark} | Range: [{start_id} - {start_id + batch - 1}]")
                new_results = test_batch_questions(llm,subtest_data,start_id,start_id + batch - 1,grpo_memory.format_memories(),test_sys,test_user_template,judge_sys,judge_user_template,times,logger,max_workers)
                success_memory,fail_memory = extract_memory(llm,new_results,memory_prompts)
                grpo_memory.process_new_memory(success_memory)
                grpo_memory.process_new_memory(fail_memory)
                logger.debug(f"📝 Raw Success Insight: {success_memory[:100]}...")
                logger.debug(f"📝 Raw Failure Insight: {fail_memory[:100]}...")
                grpo_prompt = grpo_prompt_template.render(memories = grpo_memory.format_memories())
                logger.info("🤖 Optimizing memory policy via LLM reasoning (GRPO)...")
                response_opt_memory = invoke_with_retry(
                    llm,
                    {
                        "role":"user",
                        "content":grpo_prompt,
                    }
                )
                grpo_memory.process_opt_memory(response_opt_memory)
                logger.info("✨" + "-" * 20 + " Updated Global Memory " + "-" * 20 + "✨")
                logger.info(f"\n{grpo_memory.format_memories()}")
                logger.info("✨" + "-" * 60 + "✨")
                start_id += batch
                correct_num += len(new_results["success"])
                false_num += len(new_results["failure"]["answer"])
                parse_error_num += len(new_results["failure"]["parse"])
                if start_id >= 6: break
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

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger_path = handler.baseFilename
        meta_file =logger_path[0:-3] + "json"
        with open(meta_file,"w",encoding = "utf-8") as f:
            json.dump(metadata,f,indent=4)
        logger.info(f"✅ Summary saved to {meta_file}")
    except Exception as e:
        print(f"💥 Final Save Failed: {e}")

if __name__ == "__main__":
    main()