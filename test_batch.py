import json
import yaml
import logging
import argparse

import evaluation

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from langchain_openai import ChatOpenAI

from models.llm import get_llm,invoke_with_retry
from utils.process_benchmark import process_benchmark
from utils.set_logger import setup_logger,format_log,set_filehandler

DEFAULT_MODEL_NAME = "minimax-2.5"
DEFAULT_BENCHMARK = "crwd_meta"
MAX_NUM = 20

HUMAN_MEMORY = '''
Below are some memory items that I accumulated from past interaction from the environment
that may be helpful to solve the task.You can use it when you feel it is relevant.In each step,
please discuss if you want to use each memory item or not.
# Memory Item
## Title: Atomic verification before option selection
## Description: Reasoning
## Content: Always follow this workflow: extract all available evidence, decompose each option into atomic propositions, verify every proposition against the evidence, and select only the options whose propositions are all supported. Example: If one option contains three claims and only two are supported, reject it.
'''

def test_single_question(llm,data,test_class,test_sys,test_user_template):
    problem_report = test_class.create_prompt(data)
    test_user = problem_report + '\n' + test_user_template
    
    response = invoke_with_retry(
        llm,
        {"role":"user","content":test_user},
        message_history = [{"role":"system","content":test_sys}]
    )
    return response,problem_report

def multitest_single_question(llm,data,test_class,test_sys,test_user_template,times,max_workers = 10):
    responses = []
    problem_report = None

    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {
            executor.submit(test_single_question,llm,data,test_class,test_sys,test_user_template):id
            for id in range(times)
        }
        for future in as_completed(futures):
            response,problem_report = future.result()
            responses.append(response)
    
    corrects = []
    similarities = []
    for response in responses:
        correct,similarity = test_class.check_answer(response,data)
        corrects.append(correct)
        similarities.append(similarity)

    return (responses,corrects,similarities,problem_report)

def test_batch_questions(llm,test_data,test_class,start_id,end_id,test_sys,test_user_template,times,logger,max_workers = 10):
    results = {
        "success":[],
        "failure":{"answer":[],"parse":[]}
    }
    similarity_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(multitest_single_question,llm,test_data[id],test_class,test_sys,test_user_template,times,max_workers):id
            for id in range(start_id,end_id + 1)
        }
        
        for future in as_completed(futures):
            responses,corrects,similarities,problem_report = future.result()
            similarity_results.extend(similarities)

            id = futures[future]
            test_data[id]["question"] = problem_report
            for response,correct,similarity in zip(responses,corrects,similarities):
                logger_info = format_log(test_data[id], [response], correct, jaccard_similarity = similarity)
                logger.info(logger_info)
                status = {
                    "question":problem_report,
                    "options":test_data[id]["options"],
                    "response":response
                }
                if correct == 1:
                    results["success"].append(status)
                elif correct == 0 :
                    results["failure"]["answer"].append(status)
                else:
                    results["failure"]["parse"].append(status)
    return results,similarity_results

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
    bench = args.benchmark
    times = args.times
    test_prompt_path = "configs/test_template.yml"
    max_workers = args.max_workers
    batch = args.batch
    max_tokens = 8192
    temperature = 0.7

    logger: logging.Logger
    logger,logger_name,log_dir = setup_logger("raw_test",model_name = model,benchmark_name = bench)
    set_filehandler(logger,log_dir,"setup_info")
    logger.info("="*20 + " 🛠️  System Initialization " + "="*20)
    try:
        llm = get_llm(model_name = model,temperature = temperature,max_tokens=max_tokens)
        logger.info(f"🤖 Model loaded: [ {model} ] (Temp: {temperature}) (Max_tokens = {max_tokens})")
        
        evaluation.benchmark.Benchmark.register_benchmark(evaluation.malware_analysis.MalwareAnalysisBenchmark)
        logger.info("📍 Benchmark registry is fully initialized. All discovered subclasses have been loaded and registered successfully")
        
        test_dir = Path("dataset/" + bench)
        test_data = process_benchmark(test_dir)
        sub_task_count = len(test_data)
        total_questions = sum(len(v) for v in test_data.values())
        logger.info(f"📊 Dataset loaded: '{bench}' | Sub-tasks: {sub_task_count} | Total Questions: {total_questions}")
        
        with open(test_prompt_path,"r",encoding = "utf-8") as f:
            prompts = yaml.safe_load(f)
        test_sys = prompts["test_system_template"]
        test_user_template = prompts["test_user_template"]

        # 1. 准备要打印的 Prompt 字典，方便统一记录
        all_prompts = {
            "test_sys": test_sys,
            "test_user_template_raw": prompts["test_user_template"],
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
    similarities = []
    batch_id = 0
    try :
        for sub_benchmark,subtest_data in test_data.items():  
            print(sub_benchmark)
            if not str(sub_benchmark).startswith("malware"): continue

            cls = evaluation.benchmark.Benchmark._registered_benchmarks[sub_benchmark]
            config = evaluation.benchmark.BenchmarkConfig(
                test_cases_path = test_dir / sub_benchmark,
                truncate_input = True,
                stat_path = None,
                input_modality = None,
            )
            test_class = cls(config)
            logger.info("💥 Initialze benchmark class instance")
          
            start_id = 0 
            while start_id < len(subtest_data):
                set_filehandler(logger,log_dir,f"batch_{batch_id}")
                logger.info("="*30 + " 🚀 Starting New Batch " + "="*30)
                logger.info(f"📍 Sub-benchmark: {sub_benchmark} | Range: [{start_id} - {start_id + batch - 1}]")
                
                new_results,new_similarities = test_batch_questions(llm,subtest_data,test_class,start_id,min(start_id + batch - 1,len(subtest_data) - 1),test_sys,test_user_template,times,logger,max_workers)
                similarities.extend(new_similarities)
                start_id += batch
                correct_num += len(new_results["success"])
                false_num += len(new_results["failure"]["answer"])
                parse_error_num += len(new_results["failure"]["parse"])
                batch_id += 1
                if start_id >= 100: break
    except KeyboardInterrupt:
        logger.warning("🛑 User interrupted the process. Saving current metadata...")
    except Exception as e:
        logger.error(f"💥 Unexpected system crash: {e}", exc_info=True)
            
    avg_similarity = sum(similarities) / len(similarities) if len(similarities) > 0 else 0
    
    try:
        metadata = {
            "model":model,
            "benchmark":bench,
            "times":times,
            "max_workers":max_workers,
            "correct_num":correct_num,
            "false_num":false_num,
            "parse_error_num":parse_error_num,
            "avg_jaccard_similarity":avg_similarity,
        }

        meta_file =f"{log_dir}/{logger_name}.json"
        with open(meta_file,"w",encoding = "utf-8") as f:
            json.dump(metadata,f,indent=4)
        logger.info(f"✅ Summary saved to {meta_file}")
    except Exception as e:
        print(f"💥 Final Save Failed: {e}")

if __name__ == "__main__":
    main()
