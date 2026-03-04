import logging
from pathlib import Path
from datetime import datetime


def setup_logger(file_name:str,model_name: str = "DeepSeek-V3.1-s", benchmark_name: str = "wmdp") -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"{benchmark_name}_{model_name}_{timestamp}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 避免在同一进程中重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 文件输出
    log_dir = Path("logs") / benchmark_name / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{file_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logger initialized. log_file=%s", log_file_path.as_posix())
    return logger


def main():
    # 使用默认参数测试
    logger = setup_logger("test")

    logger.info("This is an INFO message from main().")
    logger.warning("This is a WARNING message from main().")
    logger.error("This is an ERROR message from main().")


if __name__ == "__main__":
    main()