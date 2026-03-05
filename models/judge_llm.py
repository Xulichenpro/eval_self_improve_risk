from langchain_openai import ChatOpenAI
from models.llm import invoke_with_retry

def judge(llm:ChatOpenAI,responses:list[str],judge_sys_prompt:str,judge_user_prompt:str) -> list[str]:
    response = invoke_with_retry(
        llm,
        {"role":"user","content":judge_user_prompt},
        [{"role":"system","content":judge_sys_prompt}]
    )
    responses.append(response)
    return responses