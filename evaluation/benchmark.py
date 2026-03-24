from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type
from langchain_openai import ChatOpenAI
from pathlib import Path

@dataclass
class BenchmarkConfig:
    test_cases_path:Path 
    stat_path: Optional[Path] = None
    truncate_input: bool = False
    input_modality: Optional[str] = None


class Benchmark:
    _registered_benchmarks: Dict[str, Type["Benchmark"]] = {}

    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        self.test_cases_path = config.test_cases_path
        self.stat_path: Optional[Path] = config.stat_path
        

    @classmethod
    def register_benchmark(cls, benchmark_class: Type["Benchmark"]) -> None:
        """
        Registers all the benchmark kinds with the benchmark factory.
        """
        for benchmark_kind in benchmark_class.return_kind():
            cls._registered_benchmarks[benchmark_kind] = benchmark_class

    @classmethod
    def create_instance(
        cls, benchmark_kind: str, *args: Any, **kwargs: Any
    ) -> Benchmark:
        if benchmark_kind in cls._registered_benchmarks:
            benchmark_class: Type[Benchmark] = cls._registered_benchmarks[
                benchmark_kind
            ]
            # pyre-ignore [45]: This function is only used to create instances of subclasses of `Benchmark`, whose abstract methods are implemented
            return benchmark_class(*args, **kwargs)
        raise ValueError(
            f"Unknown benchmark kind: {benchmark_kind}, the registered benchmarks are: {list(cls._registered_benchmarks.keys())}"
        )

    @classmethod
    def return_kind(cls) -> list[str]:
        """
        Returns the kind(s) of benchmark that is supported. One benchmark may support multiple kinds.
        """
        raise NotImplementedError(
            "Each benchmark must implement this method to retun the kind for run script."
        )

    def extract_content_in_code_blocks(self, input: str) -> list[str]:
        # Using regular expression to find content between code blocks ```
        return re.findall(r"```(.*?)```", input, re.DOTALL)
