1. 记忆检索，embedding model -> top_k
    - 把所有memory加在prompt
2. 记忆提取
    - 剔除judge_model(tem = 0.0)，直接给出答案
    - 分开成功轨迹与失败轨迹
    - tem = 1.0
3. 记忆合并
    - 剔除重复的部分
