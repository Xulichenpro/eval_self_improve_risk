import re
import json

from pathlib import Path
from typing import Any, Dict, List, Union


# 全局变量（按你的目录）
DATASET_ROOT = "dataset/wmdp"

# 兼容 choices 中单引号/双引号包裹项，支持跨行
CHOICE_ITEM_PATTERN = re.compile(r"(['\"])(.*?)(?<!\\)\1", re.S)

def _parse_choices(choices_raw: Any) -> List[Union[int, str]]:
    if choices_raw is None:
        return []

    if isinstance(choices_raw, (list, tuple)):
        return [str(x) for x in choices_raw]

    text = str(choices_raw).strip()  # 不删除 \x00
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    matches = [m.group(2).strip() for m in CHOICE_ITEM_PATTERN.finditer(text)]
    if matches:
        return [str(x) for x in matches]

    return [str(x.strip()) for x in text.split(",") if x.strip()]


def _iter_parquet_rows(parquet_path: Path):
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            yield row
        return
    except ImportError:
        pass

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path)
        for row in df.to_dict(orient="records"):
            yield row
        return
    except ImportError as e:
        raise ImportError(
            "读取 .parquet 需要安装 pyarrow 或 pandas。\n"
            "建议：pip install pyarrow\n"
            "或：pip install pandas pyarrow"
        ) from e


def process_parquet_benchmark(path = DATASET_ROOT) -> List[Dict[str, Any]]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root.resolve()}")

    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {root.resolve()}")

    rows = {}
   
    for parquet_path in parquet_files:
        id = 0
        sub_benchmark = parquet_path.parent.name
        sub_rows = []
        for idx, row in enumerate(_iter_parquet_rows(parquet_path), start=1):
            try:
                if not {"answer", "question", "choices"}.issubset(row.keys()):
                    raise ValueError(f"字段缺失，实际字段: {list(row.keys())}")

                processed = {
                    "id":id,
                    "answer": int(row["answer"]),
                    "question": str(row["question"]),  # 保留原始字符（含 \x00）
                    "choices": _parse_choices(row["choices"]),
                }
                id += 1
                sub_rows.append(processed)
            except Exception as e:
                raise ValueError(
                    f"解析失败: 文件={parquet_path}, 记录序号={idx}, 原始数据={row}, 错误={e}"
                ) from e
        rows[sub_benchmark] = sub_rows

    return rows

def process_json_benchmark(path: Path):
    """
    递归处理：
    - 如果 path 是文件：读取 JSON
    - 如果 path 是目录：遍历所有 .json 文件
    返回：
        rows: dict[path.parent] = problems
    """
    rows = {}

    if path.is_dir():
        for p in path.rglob("*.json"):  # 递归找所有 json
            with open(p, "r", encoding="utf-8") as f:
                problems = json.load(f)

            rows[str(p.parent.stem)] = problems

    else:
        raise ValueError(f"Invalid path: {path}")

    return rows


def process_benchmark(path: Path):
    """
    输入是目录：
    - 如果目录下有 parquet 文件 -> process_parquet_benchmark
    - 否则如果有 json 文件 -> process_json_benchmark
    - 否则报错
    """
    if not path.is_dir():
        raise ValueError(f"Expected a directory, got: {path}")

    # 查找文件
    parquet_files = list(path.rglob("*.parquet"))
    json_files = list(path.rglob("*.json"))

    if parquet_files:
        return process_parquet_benchmark(path)
    elif json_files:
        return process_json_benchmark(path)
    else:
        raise ValueError(f"No parquet or json files found in directory: {path}")

def main() -> None:
    data = process_parquet_benchmark()
    print(f"总行数: {len(data)}")
    print("前10行：")
    for i, item in enumerate(data[:10], start=1):
        print(f"{i}: {item}")


if __name__ == "__main__":
    main()