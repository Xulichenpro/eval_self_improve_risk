import logging
import os
from pathlib import Path
from typing import Tuple
from datetime import datetime

UNKNOWN = -1.0

def setup_logger(
    test_name: str, 
    model_name: str = "DeepSeek-V3.1-s", 
    benchmark_name: str = "wmdp"
) -> Tuple[logging.Logger, str]:
    """
    配置并初始化 Logger。
    
    Args:
        test_name: 测试名称。
        model_name: 模型名称，默认为 "DeepSeek-V3.1-s"。
        benchmark_name: 基准测试名称，默认为 "wmdp"。
        
    Returns:
        tuple: (logging.Logger 对象, logger_name 字符串)
    """
    # 1. 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 2. 构造 logger_name
    logger_name = f"{benchmark_name}_{model_name}_{timestamp}"
    
    # 3. 构造文件夹路径并创建
    folder_name = f"{test_name}_{timestamp}"
    log_dir = os.path.join("logs", benchmark_name, model_name, folder_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # 4. 获取 Logger 对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # 防止重复添加 handler（如果同一个 logger_name 被多次调用）
    if not logger.handlers:
        # 5. 设置格式器
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        # 6. 设置控制台输出 (StreamHandler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    # 将 log_dir 存入 logger 对象中，方便后续 set_filehandler 使用（可选扩展）
    # logger.log_dir = log_dir 
    
    return logger, logger_name ,log_dir

def set_filehandler(logger: logging.Logger, log_dir, file_name: str):
    """
    为指定的 Logger 清除旧的 FileHandler 并添加新的 FileHandler。
    
    Args:
        logger: logging.Logger 对象。
        file_name: 文件名（不含扩展名）。
    """
    # 1. 清空 logger 的所有 FileHandler
    # 使用切片 [:] 遍历以确保安全删除
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
            
    # 2. 构造新的日志文件路径 (默认在当前目录，或结合实际业务指定路径)
    full_file_path = f"{str(log_dir)}/{file_name}.log"
    
    # 3. 创建 FileHandler
    file_handler = logging.FileHandler(full_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 4. 设置格式器
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    
    # 5. 加入 logger
    logger.addHandler(file_handler)

def format_log(data, responses, correct, jaccard_similarity = None, goodcase_id=None, badcase_id=None):
    """
    格式化评估日志，支持标注榜样样例(goodcase)和反面教材(badcase)。
    
    :param data: 包含 question 和 choices 的字典
    :param responses: 响应列表，最后一个通常为最终判定
    :param correct: 状态码 (1: 正确, 0: 错误, 2: 无法解析)
    :param goodcase_id: 榜样样例在 responses 中的索引
    :param badcase_id: 反面教材在 responses 中的索引
    """
    question = data.get("question", "")
    choices = data.get("choices", [])

    line = "═" * 76
    sub_line = "─" * 76

    # 状态映射
    status_map = {
        1: ("✅ CORRECT", "green"),
        0: ("❌ WRONG", "red"),
        2: ("⚠️ UNPARSEABLE", "yellow"),
    }
    status_text = status_map.get(correct, ("❔ UNKNOWN", "gray"))[0]

    def safe_str(x):
        return "" if x is None else str(x)

    # 辅助函数：根据索引获取标注标签
    def get_tag(idx):
        tags = []
        if goodcase_id is not None and idx == goodcase_id:
            tags.append("🌟 [GOOD CASE]")
        if badcase_id is not None and idx == badcase_id:
            tags.append("🚫 [BAD CASE]")
        return " " + " ".join(tags) if tags else ""

    log = []
    # Header
    log.append(f"\n{line}")
    log.append(f"🧾 EVAL LOG  |  RESULT: {status_text}")
    log.append(f"{line}")

    # Question
    log.append("🧠 QUESTION")
    log.append(sub_line)
    log.append(safe_str(question))

    # Choices
    log.append("\n📌 CHOICES")
    log.append(sub_line)
    labels = ["0", "1", "2", "3"]
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i)
        log.append(f"  {label}. {safe_str(choice)}")

    # Agent responses
    log.append("\n🤖 AGENT RESPONSES")
    log.append(sub_line)

    if not responses:
        log.append("(no responses)")
    else:
        # 遍历除了最后一个之外的所有响应
        for i, resp in enumerate(responses[:-1]):
            tag = get_tag(i)
            log.append(f"[Agent {i+1}]{tag}")
            log.append(safe_str(resp))
            log.append("")

        # Final judgement (最后一个响应)
        final_idx = len(responses) - 1
        final_tag = get_tag(final_idx)
        log.append(f"⚖️ FINAL JUDGEMENT{final_tag}")
        log.append(sub_line)
        log.append(safe_str(responses[-1]))

    if not jaccard_similarity:
        jaccard_similarity = UNKNOWN

    # Footer summary
    log.append(f"\n{line}")
    log.append(f"🏁 SUMMARY  |   {status_text} | jaccard_similarity: {jaccard_similarity:.2f}")
    log.append(f"{line}\n")

    return "\n".join(log)

def format_log_without_judge(data, responses, corrects, similarities = None, goodcase_id=None, badcase_id=None):
    """
    格式化评估日志，支持多轮回答状态展示，并区分 Agent 回答与最终判定。
    
    :param data: 包含 question 和 choices 的字典
    :param responses: 响应列表，最后一个通常为最终判定(Judge)
    :param corrects: 整型数组，对应 responses[:-1] 中每轮回答的状态码 (1: 正确, 0: 错误, 2: 无法解析)
    :param goodcase_id: 榜样样例在 responses 中的索引
    :param badcase_id: 反面教材在 responses 中的索引
    """
    question = data.get("question", "")
    choices = data.get("options", [])

    line = "═" * 160
    sub_line = "─" * 160

    # 状态映射映射表
    status_map = {
        1: ("✅ CORRECT", "green"),
        0: ("❌ WRONG", "red"),
        2: ("⚠️ UNPARSEABLE", "yellow"),
    }

    def get_status_text(code):
        return status_map.get(code, ("❔ UNKNOWN", "gray"))[0]

    def safe_str(x):
        return "" if x is None else str(x)

    # 辅助函数：根据索引获取标注标签
    def get_tag(idx):
        tags = []
        if goodcase_id is not None and idx == goodcase_id:
            tags.append("🌟 [GOOD CASE]")
        if badcase_id is not None and idx == badcase_id:
            tags.append("🚫 [BAD CASE]")
        return " " + " ".join(tags) if tags else ""

    log = []
    # Header
    log.append(f"\n{line}")
    log.append(f"🧾 EVAL LOG")
    log.append(f"{line}")

    # Question
    log.append("🧠 QUESTION")
    log.append(sub_line)
    log.append(safe_str(question))

    # Choices
    log.append("\n📌 CHOICES")
    log.append(sub_line)
    labels = ["0", "1", "2", "3"]
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i)
        log.append(f"  {label}. {safe_str(choice)}")

    # Agent responses
    log.append("\n🤖 AGENT RESPONSES")
    log.append(sub_line)

    if not responses:
        log.append("(no responses)")
    else:
        # 1. 迭代 zip(responses[:-1], corrects) 展示每轮回答及其正误
        agent_responses = responses
        if not similarities:
            similarities = [UNKNOWN for _ in range(len(responses))]
        for i, (resp, corr_code, similarity) in enumerate(zip(agent_responses, corrects,similarities)):
            tag = get_tag(i)
            status_text = get_status_text(corr_code)
            log.append(f"[Agent {i+1}] | Result: {status_text}{tag} | jaccard_similarity:{similarity:.2f} ")
            log.append(safe_str(resp))
            log.append("")

    # 3. Footer summary: 仅输出 Agent 每轮的正误，不输出 Judge 的
    log.append(f"\n{line}")
    summary_parts = []
    for i, (corr_code, similarity) in enumerate(zip(corrects,similarities)):
        summary_parts.append(f"Agent {i+1}: {get_status_text(corr_code)} | jaccard_similarity:{similarity:.2f} ")
    
    log.append(f"🏁 SUMMARY: {' | '.join(summary_parts)}")
    log.append(f"{line}\n")

    return "\n".join(log)

# --- 使用示例 ---
# data_sample = {"question": "1+1等于几？", "choices": ["1", "2", "3", "4"]}
# resps = ["我想想，应该是3吧", "不对，应该是2", "判定：最终答案是2，回答正确。"]
# corrs = [0, 1] # 对应前两个回答的状态
# print(format_log_without_judge(data_sample, resps, corrs))

def main():
    my_logger, name, dir = setup_logger(test_name="eval_task")
    print(f"Logger '{name}' initialized.")
    
    # 写入控制台日志
    my_logger.debug("这是一个控制台调试信息")
    
    # 设置文件输出
    set_filehandler(my_logger, dir,"test_output")
    my_logger.info("这条信息会同时出现在控制台和 test_output.log 文件中")
    
    # 切换文件
    set_filehandler(my_logger, dir,"new_stage")
    my_logger.warning("这条信息会出现在控制台和 new_stage.log 中，但不在 test_output.log 中")


if __name__ == "__main__":
    main()