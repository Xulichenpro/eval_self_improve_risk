import re
import json
from typing import List, Dict, Set
from jinja2 import Template

class Memory:
    def __init__(self):
        """
        初始化方法
        把memories初始化为[]
        """
        self.memories: List[Dict[str, str]] = []

    def _parse_new_memories(self, response: str) -> List[Dict[str, str]]:
        """
        辅助输入处理函数1
        解析Markdown格式的response
        提取Title, Description, Content组成字典列表
        """
        new_memories = []
        # 使用正则表达式匹配Markdown块中的内容
        # (?s) 即 re.DOTALL，使 . 能够匹配换行符
        pattern = r"#\s*Memory Item.*?\n##\s*Title:\s*(.*?)\n##\s*Description:\s*(.*?)\n##\s*Content:\s*(.*?)(?=\n)"
        
        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            memory_dict = {
                "title": match.group(1).strip(),
                "description": match.group(2).strip(),
                "content": match.group(3).strip()
            }
            new_memories.append(memory_dict)
            
        return new_memories

    def _parse_operations(self, response: str) -> List[Dict]:
        """
        辅助输入处理函数2
        从response中解析JSON数组
        并将相关id字段转换为指定的整数或整数列表格式
        """
        operations = []
        
        # 尝试从字符串中截取JSON数组部分 (以 [ 开头, ] 结尾)
        json_match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
        json_str = json_match.group(0) if json_match else response
        
        try:
            # 清理可能导致解析失败的尾随逗号 (LLM有时会生成尾随逗号)
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            raw_ops = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return operations

        # 将解析出来的字典，进行类型转换
        for op in raw_ops:
            parsed_op = op.copy()
            option = parsed_op.get("option")
            
            if option == "modify" and "modified_id" in parsed_op:
                parsed_op["modified_id"] = int(parsed_op["modified_id"])
            elif option == "merge" and "merged_id" in parsed_op:
                # 转换列表中的每一个元素为int
                parsed_op["merged_id"] = [int(i) for i in parsed_op["merged_id"]]
            elif option == "delete" and "deleted_id" in parsed_op:
                parsed_op["deleted_id"] = int(parsed_op["deleted_id"])
                
            operations.append(parsed_op)

        return operations

    def modify(self, id: int, modified_experience: Dict[str, str]):
        """
        modify函数
        用传入的字典替换指定位置的memory
        """
        if 0 <= id < len(self.memories):
            self.memories[id] = modified_experience
        else:
            print(f"修改失败: 索引 {id} 超出范围。")

    def merge(self, merged_experience: Dict[str, str]):
        """
        merge函数
        将合并后的经验放在memories列表的末尾
        """
        self.memories.append(merged_experience)

    def delete(self, ids: Set[int]):
        """
        delete函数
        删除集合中所有指定的id对应的memory
        需要将id降序排列后删除否则会因为列表长度变化导致索引偏移引发错误。
        """
        # 降序排序并依次删除
        for idx in sorted(list(ids), reverse=True):
            if 0 <= idx < len(self.memories):
                del self.memories[idx]
            else:
                print(f"删除跳过: 索引 {idx} 不存在。")

    def format_memories(self) -> str:
        """
        输出函数
        将self.memories格式化为要求的Markdown字符串并输出
        """
         # 定义排序优先级
        priority = {
            "Structure": 0,
            "Reasoning": 1,
            "Knowledge": 2
        }

        # 排序（未知类型默认放最后）
        self.memories = sorted(
            self.memories,
            key=lambda mem: priority.get(mem.get("description", ""), 999)
        )

        result_lines = []
        for i, mem in enumerate(self.memories):
            block = (
                f"# Memory Item {i}\n"
                f"## Title: {mem.get('title', '')}\n"
                f"## Description: {mem.get('description', '')}\n"
                f"## Content: {mem.get('content', '')}\n"
            )
            result_lines.append(block)
        return "\n".join(result_lines)

    def process_new_memory(self, response_new_memory: str):
        """
        综合处理函数1
        解析新增的memory内容并追加到列表中
        """
        new_memories = self._parse_new_memories(response_new_memory)
        self.memories.extend(new_memories)

    def process_opt_memory(self, response_opt_memory: str):
        """
        综合处理函数2
        解析操作指令并按顺序执行修改、合并和标记删除，最后统一执行删除
        """
        if not response_opt_memory: return
        
        operations = self._parse_operations(response_opt_memory)
        deleted_ids: Set[int] = set()
        
        for op in operations:
            option = op.get("option")
            
            if option == "modify":
                modified_experience = {
                    "title": op.get("title", ""),
                    "description": op.get("description", ""),
                    "content": op.get("content", "")
                }
                self.modify(op.get("modified_id"), modified_experience)
                
            elif option == "merge":
                merged_experience = {
                    "title": op.get("title", ""),
                    "description": op.get("description", ""),
                    "content": op.get("content", "")
                }
                self.merge(merged_experience)
                # 把涉及被合并的id加入删除集合
                deleted_ids.update(op.get("merged_id", []))
                
            elif option == "delete":
                # 单独的删除操作直接加入删除集合
                deleted_ids.add(op.get("deleted_id"))
                
        # 统一执行删除操作
        self.delete(deleted_ids)

# ================= 测试代码 (按需使用) =================
if __name__ == "__main__":
    memory_manager = Memory()
    
    # 模拟新增记录测试
    mock_new_response = """
# Memory Item  
## Title: Python Basics  
## Description: Learned how to parse markdown in python.  
## Content: RegEx is very powerful for extracting semi-structured text. 

# Memory Item  
## Title: Error Handling  
## Description: Handling JSON decoding.  
## Content: Always wrap json.loads in try-except block to handle dirty strings. 
    """
    memory_manager.process_new_memory(mock_new_response)
    print("【新增后的 Memories】:")
    print(memory_manager.format_memories())

    # 模拟操作记录测试 (修改0，把0和1合并作为新条目并删除0和1)
    mock_opt_response = """
Here is the operations array:
```json
[
    {
        "option": "modify",
        "modified_id": "0",
        "title": "Python Basics (Modified)",
        "description": "Learned regex and JSON parsing.",
        "content": "RegEx and json.loads are essential."
    },
    {
        "option": "merge",
        "merged_id": ["0", "1"],
        "title": "Python Programming Skills",
        "description": "Comprehensive summary of Python basic parsings.",
        "content": "Mastering RegEx for string extraction and JSON libraries for data structure processing prevents runtime errors."
    }
]
```
    """
    memory_manager.process_opt_memory(mock_opt_response)
    print("【执行操作后的 Memories】:")
    print(memory_manager.format_memories())
    template = '''
An agent system is provided with a set of memories and has tried to solve problems multiple times. 
From the reflections, we have got some raw memories. 
Your task is to collect and think for the final memory revision plan. 
Each final memory must satisfy the following requirements:
1. It must be clear, generalizable lessons for this case, with no more than 40 words
2. Begin with **general** background with several words in the memory
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating saying similar memory in multiple different memories
<memory> 
{{memories}} 
</memory>

Please think for how to update existing experiences,
You have three update options: [modify, merge, delete]
- modify: You can modify current experiences to make it helpful
- merge: You can merge some similar experiences into a more general forms to reduce duplication
The input format:
```  markdown
# Memory Item i
## Title: <the title of the memory item>  
## Description: <one sentence summary of the memory item>  
## Content: <1-3 sentences describing the insights learned to successfully accomplishing the task> 
```

Your output should follow the format shown below exactly:

```json
[
    {
        "option": "modify",
        "modified_id": "2", //An int.For Memory Item i,its id is i(an int).
        "title":"modified title",
        "description":"modified description",
        "content":"modified content",
        "think":"your thought",
    },
    {
        "option": "merge",
        "merged_id": ["1", "3", "4"],  //An int list.For Memory Item i,its id is i(an int).
        "title":"merged title",
        "description":"merged description",
        "content":"merged content",
        "think":"your thought",
    },
    {
        "option": "delete",
        "deleted_id": "5",//an int
        "think":"your thought",
    }, 
]
```
    '''
    template = Template(template)
    print(template.render(memories = memory_manager.format_memories()))
