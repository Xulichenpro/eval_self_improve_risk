import os
import re

def analyze_logs(directory_path):
    if not os.path.exists(directory_path):
        print(f"错误: 找不到目录 '{directory_path}'")
        return

    # 统计项初始化
    stats = {
        "same_correct": 0,    # 3次回答相同且最终正确
        "same_wrong": 0,      # 3次回答相同但最终错误
        "diff_correct": 0,    # 3次回答不同/解析失败且最终正确
        "diff_wrong": 0,      # 3次回答不同/解析失败且最终错误
        "parse_error":0,
    }

    # 正则表达式
    # 匹配 <answer> 标签内容
    ans_regex = re.compile(r'<answer>\s*(\d+)\s*</answer>', re.IGNORECASE)
    # 匹配 SUMMARY 状态
    sum_regex = re.compile(r'🏁 SUMMARY\s*\|\s*(?:✅|❌)?\s*(CORRECT|WRONG)', re.IGNORECASE)
    # 模块分割符：匹配类似 "2026-03-13 19:48:28 | INFO |" 的行
    block_splitter = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \|', re.MULTILINE)

    log_files = [f for f in os.listdir(directory_path) if f.endswith('.log')]
    
    if not log_files:
        print("在该目录下未找到 .log 文件。")
        return

    total_blocks_processed = 0

    for filename in log_files:
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # 使用正则切分模块
            # split 会保留分割点之后的内容，我们要过滤掉第一个可能的空字符串
            parts = block_splitter.split(content)
            # 如果文件开头就是时间戳，第一个元素是空的
            blocks = [p for p in parts if "EVAL LOG" in p] 

            for block in blocks:          
                # 1. 提取 Agent 答案
                agent_answers = ans_regex.findall(block)
                
                # 2. 提取 Summary 结果
                summary_match = sum_regex.search(block)
                is_correct = False
                if summary_match:
                    is_correct = (summary_match.group(1).upper() == "CORRECT")
                    total_blocks_processed += 1
                else :
                    continue
                # 3. 逻辑判定
                # 判断 3 个 Agent 是否一致且都解析到了
                is_consistent = (len(agent_answers) == 3 and len(set(agent_answers)) == 1)
                diff = (len(agent_answers) == 3 and len(set(agent_answers)) >= 1)

                if is_consistent:
                    if is_correct:
                        stats["same_correct"] += 1
                    else:
                        stats["same_wrong"] += 1
                elif diff:
                    if is_correct:
                        stats["diff_correct"] += 1
                    else:
                        stats["diff_wrong"] += 1
                else:
                    stats["parse_error"] +=1

    # 打印结果
    print("\n" + "="*60)
    print(f"📊 统计报告 (共处理 {len(log_files)} 个文件，包含 {total_blocks_processed} 个评估模块)")
    print("-" * 60)
    print(f"✅ 1. Agent 3次回答相同，最终正确 (CORRECT):      {stats['same_correct']}")
    print(f"❌ 2. Agent 3次回答相同，最终错误 (WRONG):        {stats['same_wrong']}")
    print(f"⚠️ 3. Agent 回答不同，最终正确 (CORRECT): {stats['diff_correct']}")
    print(f"🚫 4. Agent 回答不同，最终错误 (WRONG):   {stats['diff_wrong']}")
    print(f"🤖 5. Agent 回答中有解析失败的:   {stats['parse_error']}")
    print("="*60 + "\n")

