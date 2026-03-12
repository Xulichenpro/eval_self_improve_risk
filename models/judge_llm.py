import re
from typing import Optional
from jinja2 import Template
from langchain_openai import ChatOpenAI
from models.llm import invoke_with_retry

def judge(
    llm: ChatOpenAI, 
    responses: list[str], 
    judge_sys_prompt: str, 
    judge_user_prompt: str,
    choice_prompts: dict[str, Template]
) -> tuple[list[str], Optional[int], Optional[int]]:
    """
    根据 LLM 的多个响应进行判断，并根据一致性选择后续处理模板。
    """

    # --- 1. 执行初始逻辑：请求模型并记录响应 ---
    # 根据 invoke_with_retry 的定义，它会将 message_json 添加到 message_history 中并调用
    # 第一次调用时，我们将 sys_prompt 作为历史传入
    initial_response = invoke_with_retry(
        llm,
        {"role": "user", "content": judge_user_prompt},
        [{"role": "system", "content": judge_sys_prompt}]
    )
    responses.append(initial_response)

    # --- 2. 检查 responses 中的答案一致性 ---
    answer_pattern = r"<answer>(.*?)</answer>"
    extracted_answers = []
    
    for r in responses[:-1]:
        match = re.search(answer_pattern, r, re.DOTALL)
        if match:
            try:
                # 尝试转换为 int
                val = int(match.group(1).strip())
                extracted_answers.append(val)
            except ValueError:
                # 如果内容不是有效的整数，视为解析失败
                pass

    # 判断逻辑：是否所有答案都解析成功，且所有解析出的 int 均相同
    is_consistent = False
    if len(extracted_answers) == len(responses) and len(responses) > 0:
        if all(x == extracted_answers[0] for x in extracted_answers):
            is_consistent = True

    # --- 3. 选择并渲染 Template ---
    selected_template = choice_prompts["best_of_n"] if is_consistent else choice_prompts["two_different_trajs"]
    
    # 参数名为 reponse (按需求文档拼写)，内容为换行连接的 responses
    #combined_responses_str = "\n".join(responses[:-1])
    rendered_prompt = selected_template.render(reponses=responses[:-1])

    # --- 4. 构建历史记录并再次请求 ---
    # 把前一次 judge 的 prompt (system + user) 及其 response 放入历史
    msg_history = [
        {"role": "system", "content": judge_sys_prompt},
        {"role": "user", "content": judge_user_prompt},
        {"role": "assistant", "content": initial_response}
    ]

    # 第二次请求模型
    second_response = invoke_with_retry(
        llm,
        {"role": "user", "content": rendered_prompt},
        msg_history
    )
    
    # 将第二次请求的结果也放入 responses (可选，根据您的返回需求)
    # responses.append(second_response)

    # --- 5. 解析 goodcase 和 badcase ---
    # 处理用户提到的 <\goodcase> 反斜杠变体
    goodcase_pattern = r"<goodcase>\s*(\d+)\s*</goodcase>"
    # 处理无闭合标签或普通闭合标签的 badcase
    badcase_pattern = r"<badcase>\s*(\d+)\s*</badcase>"

    goodcase_id = None
    badcase_id = None

    g_match = re.search(goodcase_pattern, second_response, re.IGNORECASE)
    if g_match:
        goodcase_id = int(g_match.group(1))

    b_match = re.search(badcase_pattern, second_response, re.IGNORECASE)
    if b_match:
        badcase_id = int(b_match.group(1))

    return responses, second_response, goodcase_id, badcase_id