import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import logging
from pathlib import Path
import yaml
import re
import json

from jinja2 import Template
from typing import Any
from langchain_openai import ChatOpenAI

from models.llm import get_llm,invoke_with_retry
from models.judge_llm import judge
from utils.process_benchmark import process_parquet_benchmark
from utils.set_logger import setup_logger,format_log

DEFAULT_MODEL_NAME = "DeepSeek-V3.1-s"
DEFAULT_BENCHMARK = "wmdp"
MAX_NUM = 20

def test_single_question(llm,data,memories,test_sys,test_user_template):
    test_user = test_user_template.render(
        question = data["question"],
        choices = data["choices"],
        memory = "/n".join(memories)
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
    parser.add_argument("--batch",type=int,default=5,help = "max workers for ThreadPoolExecutors")
    args = parser.parse_args()
    args = parser.parse_args()

    model = args.model
    benchmark = args.benchmark
    times = args.times
    test_prompt_path = "configs/test_template.yml"
    memory_prompt_path = "configs/reasoning_bank_template.yml"
    max_workers = args.max_workers
    batch = args.batch

    logger:logging.Logger = setup_logger("test_with_memory",model_name = model,benchmark_name = benchmark)

    try:
        llm = get_llm(model_name = model,temperature = 0.2)
        logger.info("setup the test model")
        test_data = process_parquet_benchmark("dataset/" + benchmark)
        logger.info("setup the test problem")
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
        logger.info("setup prompt template")
    except Exception as e:
        logger.warning("Setup fail!")
    
    memories = []
    start_id = 0
    results = {}
    while start_id < 15 :
        new_results = test_batch_questions(llm,test_data,start_id,start_id + batch - 1,memories,test_sys,test_user_template,judge_sys,judge_user_template,times,logger,max_workers)
        new_memories = extract_memory(llm,new_results,memory_prompts)
        logger.info("🧠 Get new memoryies!\n" + "\n".join(new_memories))
        memories.extend(new_memories)
        start_id += batch
        results = results | new_results #TODO:字典嵌套合并
    
    correct_num, false_num, parse_error_num = len(results["success"]),len(results["failure"]["answer"]) ,len(results["failure"]["parse"])

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


if __name__ == "__main__":
    main()