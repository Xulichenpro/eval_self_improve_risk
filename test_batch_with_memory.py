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
from utils.set_logger import setup_logger,format_log,set_filehandler

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

def multitest_single_question(llm,data,memories,test_sys,test_user_template,judge_sys,judge_user_template,choice_prompts,times,max_workers = 10):
    responses = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = {
            executor.submit(test_single_question,llm,data,memories,test_sys,test_user_template):id
            for id in range(times)
        }
        for future in as_completed(futures):
            responses.append(future.result())
    judge_user_prompt = judge_user_template.render(data = data,times = times,responses = responses,memory = memories)
    responses, judge_response,goodcase_id, badcase_id = judge(llm,responses,judge_sys,judge_user_prompt,choice_prompts)
    
    correct = 2
    match = re.search(r'"answer"\s*:\s*(\d+)', responses[-1])
    if match:
        try :
            answer = int(match.group(1).strip())
            if answer == data["answer"]:
                correct = 1
            else:
                correct = 0
        except:
            correct = 2
    return (responses,judge_response,correct,goodcase_id,badcase_id)

def test_batch_questions(llm,test_data,start_id,end_id,memories,test_sys,test_user_template,judge_sys,judge_user_template,choice_prompts,times,logger,max_workers = 10):
    results = {
        "success":{},
        "failure":{"answer":{},"parse":{}}
    }
    model_results = {
        "success":{},
        "failure":{}
    }
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(multitest_single_question,llm,test_data[id],memories,test_sys,test_user_template,judge_sys,judge_user_template,choice_prompts,times,max_workers):id
            for id in range(start_id,end_id + 1)
        }
        
        for future in as_completed(futures):
            responses,judge_response,correct,goodcase_id,badcase_id = future.result()
            id = futures[future]
            full_judge = responses[-1] + '\n' + judge_response
            logger_info = format_log(test_data[id], responses[:-1] + [full_judge], correct,goodcase_id,badcase_id)
            logger.info(logger_info)
            id = futures[future]
            status = {
                    "question":test_data[id]["question"],
                    "choices":test_data[id]["choices"],
                    "responses":responses,
                    "judge_response":judge_response,
                }
            if correct == 1:
                results["success"][id] = status
            elif correct == 0 :
                results["failure"]["answer"][id] = status 
            else:
                results["failure"]["parse"][id] = status
            if goodcase_id:
                model_results["success"][id] = {
                    "question":test_data[id]["question"],
                    "choices":test_data[id]["choices"],
                    "response":responses[goodcase_id],
                }
                 
            if badcase_id:
                model_results["failure"][id] = {
                    "question":test_data[id]["question"],
                    "choices":test_data[id]["choices"],
                    "response":responses[badcase_id],
                }
    return results,model_results
#TODO: 防止超过上下文窗口，对log分块
def extract_memory(llm,results,memory_prompts):
    success_user = Template(memory_prompts["success_trajectory"]["user_template"]).render(results = results["success"])
    fail_user = Template(memory_prompts["fail_trajectory"]["user_template"]).render(results = results["failure"])
    try :
        success_memory = invoke_with_retry(
            llm,
            {"role":"user","content":success_user},
            [{"role":"system","content":memory_prompts["success_trajectory"]["sys_template"]}]
        )
    except:
        success_memory = "No success memory." 
    
    try:
        fail_memory = invoke_with_retry(
            llm,
            {"role":"user","content":fail_user},
            [{"role":"system","content":memory_prompts["fail_trajectory"]["sys_template"]}]
        )
    except:
        fail_memory = "No fail memory."
    return [success_memory,fail_memory]

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
    max_tokens = 4096
    temperature = 0.7

    logger: logging.Logger
    logger,logger_name,log_dir = setup_logger("test_with_memory",model_name = model,benchmark_name = benchmark)
    set_filehandler(logger,log_dir,"setup_info")
    logger.info("="*20 + " 🛠️  System Initialization " + "="*20)
    try:
        llm = get_llm(model_name = model,temperature = temperature,max_tokens = max_tokens)
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
        choice_prompts = {
            "best_of_n": Template(prompts["choice_prompts"]["best_of_n"]),
            "two_different_trajs":Template(prompts["choice_prompts"]["two_different_trajs"])
        }
        with open(memory_prompt_path,"r",encoding = "utf-8") as f:
            memory_prompts = yaml.safe_load(f)
        with open (grpo_prompt_path,"r",encoding = "utf-8") as f:
            grpo_prompts = yaml.safe_load(f)
        grpo_prompt_template = Template(grpo_prompts["grpo_user_template"])

        # 1. 准备要打印的 Prompt 字典，方便统一记录
        all_prompts = {
            "test_sys": test_sys,
            "test_user_template_raw": prompts["test_user_template"] + prompts["user_template_with_memory"],
            "judge_sys": judge_sys,
            "judge_user_template_raw": prompts["judege_user_template"] + prompts["user_template_with_memory"],
            "memory_prompts": memory_prompts,
            "grpo_user_template_raw": grpo_prompts["grpo_user_template"],
            "best_of_n":prompts["choice_prompts"]["best_of_n"],
            "two_different_trajs":prompts["choice_prompts"]["two_different_trajs"],
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
    
    dashboard_data = {
        "summary": {},
        "prompts": all_prompts,
        "batches": []
    }
    
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
                new_results,model_results = test_batch_questions(llm,subtest_data,start_id,start_id + batch - 1,grpo_memory.format_memories(),test_sys,test_user_template,judge_sys,judge_user_template,choice_prompts,times,logger,max_workers)
                success_memory,fail_memory = extract_memory(llm,model_results,memory_prompts)
                grpo_memory.process_new_memory(success_memory)
                grpo_memory.process_new_memory(fail_memory)
                logger.debug(f"📝 Raw Success Insight: {success_memory}...")
                logger.debug(f"📝 Raw Failure Insight: {fail_memory}...")
                logger.info(f"📝 Raw Memory Module Output: {grpo_memory.format_memories()}")#memories
                grpo_prompt = grpo_prompt_template.render(memories = grpo_memory.format_memories())
                logger.info(f"{grpo_prompt}")

                try :
                    response_opt_memory = invoke_with_retry(
                        llm,
                        {
                            "role":"user",
                            "content":grpo_prompt,
                        }
                    )
                except:
                    response_opt_memory = "Fail to get memory optimization plan."
                logger.info(f"🤖 Optimizing memory policy via LLM reasoning (GRPO)...\n {response_opt_memory}")
                grpo_memory.process_opt_memory(response_opt_memory)
                logger.info("✨" + "-" * 20 + " Updated Global Memory " + "-" * 20 + "✨")
                #logger.info(f"raw output :/n{response_opt_memory}/n/n")
                logger.info(f"\n{grpo_memory.format_memories()}")
                logger.info("✨" + "-" * 60 + "✨")
                
                correct_num += len(new_results["success"])
                false_num += len(new_results["failure"]["answer"])
                parse_error_num += len(new_results["failure"]["parse"])

                batch_questions = []

                # 提取：成功 Cases (Good Cases)
                for q_id, q_data in new_results.get("success", {}).items():
                    batch_questions.append({
                        "id": q_id, 
                        "is_good": True,
                        # ⚠️ 注意：以下 get() 里的键名("question", "options"等)，
                        # 请确保与您 q_data 里实际存储的键名一致。如果不一致请替换。
                        "question": q_data.get("question", "未知题目"), 
                        "options": q_data.get("options", []), 
                        # 如果您的系统是单次回答，可能存的是 "response"，我们可以把它包装成列表
                        "trajectories": q_data.get("responses", ["No response"]), 
                        "judge": q_data.get("judge_response", "No judge")
                    })

                # 提取：回答错误 Cases (Bad Cases - Answer)
                for q_id, q_data in new_results.get("failure", {}).get("answer", {}).items():
                    batch_questions.append({
                        "id": q_id,
                        "is_good": False,
                        "question": q_data.get("question", "未知题目"),
                        "options": q_data.get("options", []),
                        "trajectories": q_data.get("responses", ["No response"]),
                        "judge": q_data.get("judge_response", "No judge")
                    })

                # 提取：解析失败 Cases (Bad Cases - Parse Error)
                for q_id, q_data in new_results.get("failure", {}).get("parse", {}).items():
                    batch_questions.append({
                        "id": q_id,
                        "is_good": False, # 解析失败也算作 Bad Case
                        "question": q_data.get("question", "未知题目"),
                        "options": q_data.get("options", []),
                        # 解析失败通常记录了模型的原始乱码输出
                        "trajectories": q_data.get("responses", ["No response"]),
                        "judge": q_data.get("judge_response", "No judge")
                    })

                # 为了让前端展示时题目顺序不乱，我们按照题号 id 重新排序一下
                # (假设您的 q_id 可以转换为 int，如果本身是数字就不需要强转)
                try:
                    batch_questions.sort(key=lambda x: int(x["id"]))
                except ValueError:
                    pass 

                batch_record = {
                        "id": batch_id,
                        "sub_benchmark": sub_benchmark,
                        "range": f"{start_id} - {start_id + batch - 1}",
                        "questions": batch_questions,
                        "memory": {
                            "raw_success": str(success_memory),
                            "raw_fail": str(fail_memory),
                            "response_opt_memory": str(response_opt_memory),
                            "updated_global_memory": str(grpo_memory.format_memories())
                        }
                    }
                dashboard_data["batches"].append(batch_record)
                
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
    finally:

        import os
        import shutil
        
        try:
            source_html_path = "index.html"
            target_html_path = os.path.join(log_dir, "index.html")
            
            # 读取并写入的方式（完全按照您的需求）
            with open(source_html_path, "r", encoding="utf-8") as src_file:
                html_content = src_file.read()
                
            with open(target_html_path, "w", encoding="utf-8") as dest_file:
                dest_file.write(html_content)
                
            # 注意：其实在 Python 中用 shutil.copy("index.html", target_html_path) 也可以达到一模一样的效果且更简洁
            
            logger.info(f"🌐 网页文件已成功生成至: {target_html_path}")
            logger.info(f"👉 请在终端进入 {log_dir} 文件夹，并运行 'python -m http.server 8000' 来查看可视化面板。")
            
        except FileNotFoundError:
            logger.error(f"🛑 找不到源网页文件: {source_html_path}。请确保它与您的 Python 脚本在同一目录下！")
        except Exception as e:
            logger.error(f"🛑 写入 index.html 时发生错误: {e}")

        metadata = {
            "model": model,
            "benchmark": benchmark,
            "temperature": 0.2, # 写入您设定的参数
            "times": times,
            "correct_num": correct_num,
            "false_num": false_num,
            "parse_error_num": parse_error_num,
            "total_questions": total_processed_questions
        }
        dashboard_data["summary"] = metadata
        
        # 原有保存 metadata 的逻辑...
        
        # 👑 新增：保存给前端使用的 JSON 数据
        dashboard_file = f"{log_dir}/result.json"
        with open(dashboard_file, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ Dashboard Data saved to {dashboard_file}")

if __name__ == "__main__":
    main()