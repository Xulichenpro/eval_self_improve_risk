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
from utils.set_logger import setup_logger

DEFAULT_MODEL_NAME = "DeepSeek-V3.1-s"
DEFAULT_BENCHMARK = "wmdp"
MAX_NUM = 20

def test_single_question(llm,data,test_sys,test_user_template):
    test_user = test_user_template.render(
        question = data["question"],
        choices = data["choices"]
    )
    response = invoke_with_retry(
        llm,
        {"role":"user","content":test_user},
        message_history = [{"role":"system","content":test_sys}]
    )
    return response

def multitest_single_question(llm,data,test_sys,test_user_template,times,max_workers = 10):
    responses = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {
            executor.submit(test_single_question,llm,data,test_sys,test_user_template):id
            for id in range(times)
        }
        for future in as_completed(futures):
            responses.append(future.result())
    answer_pattern = r"<answer>(.*?)</answer>"
    correct = 2
    
    match = re.search(answer_pattern, responses[0], re.S)
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

def main():
    parser = argparse.ArgumentParser(description="setup the parameter for test")
    parser.add_argument("--model","-m",default = DEFAULT_MODEL_NAME,help="the model you want to test")
    parser.add_argument("--benchmark","-b",default=DEFAULT_BENCHMARK,help="the benchmark you want to test")
    parser.add_argument("--times","-t",type=int,default=1,help = "the number of times a question will be tested")
    parser.add_argument("--max_workers","-n",type=int,default=10,help = "max workers for ThreadPoolExecutors")
    args = parser.parse_args()

    model = args.model
    benchmark = args.benchmark
    times = args.times
    test_prompt_path = "configs/test_template.yml"
    max_workers = args.max_workers

    logger:logging.Logger = setup_logger("test",model_name = model,benchmark_name = benchmark)

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
        test_user_template = Template(prompts["test_user_template"])
        logger.info("setup prompt template")
    except Exception as e:
        logger.warning("Setup fail!")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(multitest_single_question,llm,test_data[id],test_sys,test_user_template,times,max_workers):id
            for id in range(len(test_data))
        }

        for future in as_completed(futures):
            responses,correct = future.result()
            if correct == 0:
                correct_info = "False"
            elif correct == 1:
                correct_info = "True"
            else:
                correct_info = "Cannot Parse"
            logger_info = "AGENT ACTION\n" + "\n".join(responses) + "\n" + f"correct : {correct_info}"
            logger.info(logger_info)
            id = futures[future]
            results[id] = {
                "agent_action":"\n".join(responses) + "\n",
                "correct":correct
            }

    correct_num, false_num, parse_error_num = (0,0,0)
    for id,status in results.items():
        if status["correct"] == 0:
            false_num += 1
        elif status["correct"] == 1:
            correct_num += 1
        else:
            parse_error_num += 1

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
