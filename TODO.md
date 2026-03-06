1. wdmp三部分分开测试
2. embedding模型查询
3. memory分块管理

# 用python写一个类，命名为Memory
## 属性
1. 列表memories，每个元素是一个二元元组(embed,text),embed是经处理的向量，text是个字典，有三个key,title,description,content
2. embedding模型：embeding_model

## 方法
1. 初始化要求
    - memories初始化为空
    - embedding模型获取
2. memory添加函数
    - 输入格式，含有多个如下所示的markdown块的字符串：
    ``` markdown
    # Memory Item  
    ## Title: <the title of the memory item>  
    ## Description: <one sentence summary of the memory item>  
    ## Content: <1-3 sentences describing the insights learned to successfully accomplishing the task> 
    ```
    - 请你解析这个字符串，把每个memory item的Title部分对应的内容用embedding_model编码，构成embed，
      把Title Description Content对应的内容解析出来放入字典text，加入memories

3. memory查询函数
    - 输入格式，含有字符串query和整数变量top_k，默认为3
    - 用embedding_model编码query为embed_query
    - 计算memories列表中每个元素的embed和embed_query的cosine距离，选出top_k个
    - 把这top_k个元素按如下格式整合为1个str输出：
    ``` markdown
    # Memory Item  
    ## Title: <the title of the memory item>  
    ## Description: <one sentence summary of the memory item>  
    ## Content: <1-3 sentences describing the insights learned to successfully accomplishing the task> 
    ```
