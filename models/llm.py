from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_fixed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yml"
DEFAULT_MODEL_NAME = "DeepSeek-V3.1-s"


def _load_model_config(config_path: Path) -> dict[str, Any]:
    """Read model configuration from yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure in {config_path}: top level must be a mapping")

    return data


def get_llm(model_name: str = DEFAULT_MODEL_NAME, config_path: Path = DEFAULT_CONFIG_PATH,temperature = 0.2,max_tokens = 4096) -> ChatOpenAI:
    """Build an invokable LangChain chat model by model name.

    Args:
        model_name: The key under configs/model.yml (default: DeepSeek-V3.1).
        config_path: Path to model.yml.

    Returns:
        A ChatOpenAI instance that supports `.invoke(...)`.
    """
    all_configs = _load_model_config(config_path)
    if model_name not in all_configs:
        available = ", ".join(sorted(all_configs.keys()))
        raise KeyError(f"Unknown model: {model_name}. Available models: {available}")

    model_cfg = all_configs[model_name] or {}
    api_key = model_cfg.get("openai_api_key")
    api_base = model_cfg.get("openai_api_base")
    actual_model = model_cfg.get("model", model_name)

    if not api_key or not api_base:
        raise ValueError(
            f"Model '{model_name}' must contain both 'openai_api_key' and 'openai_api_base' in {config_path}"
        )

    return ChatOpenAI(
        model=actual_model,
        api_key=api_key,
        base_url=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def invoke_with_retry(llm: ChatOpenAI, message_json: dict[str, str], message_history:list = []) -> str:
    """Invoke llm with retry using tenacity."""

    message_history.append(message_json)

    response = llm.invoke(message_history)
    #print(response)
    return response.content

def main() -> None:
    parser = argparse.ArgumentParser(description="Load and test LLM invocation via LangChain.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name key in configs/model.yml")
    parser.add_argument("--prompt", default="Introduce Yourself in one sentence", help="Prompt for invocation test")
    args = parser.parse_args()

    llm = get_llm(args.model)
    request_json = {"role": "user", "content": args.prompt}
    answer = invoke_with_retry(llm, request_json)
    print(answer)


if __name__ == "__main__":
    main()