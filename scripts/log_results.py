import re
from pathlib import Path

# 设置全局变量
log_dir = Path(__file__).parent.parent / "logs"

def parse_memory_log(log_path: str, start_id: int, end_id: int):
    """
    解析 memory 类型的日志，通常包含多个 Agent 的结果汇总。
    """
    target_dir = log_dir / log_path
    success_num, fail_num, parse_error_num = 0, 0, 0

    for i in range(start_id, end_id + 1):
        file_path = target_dir / f"batch_{i}.log"
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 找到所有符合条件的摘要行内容
            # 在 memory log 中，摘要行通常长这样：🏁 SUMMARY: Agent 1: ❌ WRONG | Agent 2: ❌ WRONG ...
            summaries = re.findall(r"🏁 SUMMARY: (.*?)(?:\n|$)", content)
            
            for summary in summaries:
                # 使用 count() 统计一行中出现的总次数
                success_num += summary.count("✅ CORRECT")
                fail_num += summary.count("❌ WRONG")
                parse_error_num += summary.count("⚠️ UNPARSEABLE")

    print(f"--- Memory Log Parse Result ({target_dir}) ---")
    print(f"Success: {success_num}, Fail: {fail_num}, Parse Error: {parse_error_num}, Accuracy: {success_num / (success_num + fail_num + parse_error_num)})")
    return success_num, fail_num, parse_error_num


def parse_raw_log(log_path: str, start_id: int, end_id: int):
    """
    解析 raw 类型的日志，通常包含单次判断的结果。
    """
    target_dir = log_dir / log_path
    success_num, fail_num, parse_error_num = 0, 0, 0

    for i in range(start_id, end_id + 1):
        file_path = target_dir / f"batch_{i}.log"
        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 在 raw log 中，摘要行长这样：🏁 SUMMARY  |   ❌ WRONG
            # 使用正则匹配这一行
            summaries = re.findall(r"🏁 SUMMARY\s+\|\s+(.*?)(?:\n|$)", content)
            
            for status in summaries:
                if "✅ CORRECT" in status:
                    success_num += 1
                elif "❌ WRONG" in status:
                    fail_num += 1
                elif "⚠️ UNPARSEABLE" in status:
                    parse_error_num += 1

    print(f"--- Raw Log Parse Result ({target_dir}) ---")
    print(f"Success: {success_num}, Fail: {fail_num}, Parse Error: {parse_error_num}, Accuracy: {success_num / (success_num + fail_num + parse_error_num)}")
    return success_num, fail_num, parse_error_num

# 示例用法示例 (如果需要直接运行脚本):
if __name__ == "__main__":
    # 请确保目录存在
    # log_dir.mkdir(exist_ok=True)
    # parse_memory_log("wmdp_test", 0, 5)
    parse_memory_log("wmdp/Qwen3-235B-A22B-Instruct-2507/test_without_judge_20260316_044556",0,56)
    parse_raw_log("wmdp/Qwen3-235B-A22B-Instruct-2507/raw_test_20260316_044415",0,56)