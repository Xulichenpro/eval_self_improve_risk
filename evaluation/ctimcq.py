import re

from typing import Any, cast, Dict, List, Optional, Union

from .benchmark import Benchmark,BenchmarkConfig
from utils.eval_utils import jaccard_similarity

class CTIMCP(Benchmark):
    def __init__(
            self,
            config:BenchmarkConfig,
    ):
        super().__init__(config)
        self.test_cases_path = config.test_cases_path

    @classmethod
    def return_kind(cls):
        return ["ctimcq"]
    
    def create_prompt(
        self, test_case: Dict[str, Union[int, str]],
    ) -> str:
        """Create a prompt for the given test case."""

        return test_case["Prompt"]
    
    def check_answer(self, response, data):
        answer_pattern = r"<scratchpad>.*?</scratchpad>.*?<answer>(.*?)</answer>"
        matches = list(re.finditer(answer_pattern, response, re.S))

        # 没有任何 match
        if not matches:
            return 2, 0

        extracted = None

        # 迭代寻找第一个 extracted 非空的 match
        for match in reversed(matches):
            answer_str = match.group(1).strip()
            cur_extracted = [x.upper() for x in re.findall(r"[A-Za-z]", answer_str)]
            if cur_extracted:
                extracted = cur_extracted
                break

        # 所有 match 都无法提取出有效答案
        if not extracted:
            return 2, 0

                # 正确答案
        answer = data.get("GT", None)
        answer = [x.upper() for x in re.findall(r"[A-Za-z]", answer)]
        correct = set(answer)
        similarity = jaccard_similarity(extracted, set(correct))

        # 完全相等才正确
        if set(extracted) == set(correct):
            return 1, similarity
        else:
            return 0, similarity