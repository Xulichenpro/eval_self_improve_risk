from models.llm import get_llm,invoke_with_retry

model_name = "Qwen3-235B-A22B-Instruct-2507"

test_user = '''
Please answer following question:
What's the result of 10234 * 7890?

Output format (must follow exactly):
<scrachpad>
{your_thought_here}
</scrachpad>
<answer>
{your_answer_here}
</answer>
'''
llm = get_llm(model_name=model_name,temperature=0.8)
response = invoke_with_retry(
    llm,
    {"role":"user","content":test_user}
)
print(response)