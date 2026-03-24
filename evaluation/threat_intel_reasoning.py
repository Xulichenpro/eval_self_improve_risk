import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple, Union

from .benchmark import Benchmark, BenchmarkConfig

class ThreatIntelReasoningBenchmark(Benchmark):
    """
    This class implements the threat intel reasoning SOC benchmark.
    """

    test_cases_path: Optional[Path] = None

    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        super().__init__(config)
        self.test_cases_path = config.test_cases_path
        self.input_modality: str = (
            config.input_modality if config.input_modality else "text"
        )
        modalities = ["text", "image", "text_and_image"]
        formated_modalities = ", ".join([f"'{m}'" for m in modalities])
        assert self.input_modality in modalities, (
            f"Supported input_modality values for ThreatIntelReasoningBenchmark are {formated_modalities}"
        )

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["threat_intel_reasoning"]

    def create_prompt(
        self, test_case: Dict[str, Union[int, str]]
    ) -> Tuple[str, Optional[str], Optional[List[str]]]:
        """Create a prompt for the given test case.  Returns a tuple of the text prompt and the image path."""
        question = test_case.get("question_text")
        options = test_case.get("options")
        report_id = cast(str, test_case.get("report_id"))

        assert self.test_cases_path is not None
        prompt_directory = os.path.dirname(self.test_cases_path)

        prompt = f"""

        Given the context, answer the following question: {question}. Options: {options}.
        You need to return the list of the list of the letters corresponding to the correct answers, just the letters.
        For example:["A","D"]

        """
        text_path, images = None, None

        if "text" in self.input_modality:
            text_path = prompt_directory + "/" + report_id + ".txt"
            context = ""
            with open(text_path, "r") as file:
                context = file.read()
            prompt = (
                f"""
            {context}
            """
                + prompt
            )

        if "image" in self.input_modality:
            pattern = f"{prompt_directory}/{report_id}__*__.png"
            images = glob.glob(pattern)
            if len(images) == 0:
                logging.warning(f"No images found for {report_id}")

        return (prompt, text_path, images)

    def check_answer(self, response, data):
        answer_pattern = r"</scrachpad>.*?<answer>(.*?)</answer>"
        match = re.search(answer_pattern, response, re.S)

        # 不能解析 → 返回1
        if not match:
            return 2

        # 提取答案字母
        answer_str = match.group(1).strip()
        extracted = [x.upper() for x in re.findall(r"[A-Za-z]", answer_str)]

        # 正确答案
        correct = data.get("correct_options", [])

        # 完全相等才正确
        if set(extracted) == set(correct):
            return 1
        else:
            return 0