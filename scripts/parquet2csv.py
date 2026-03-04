import pandas as pd
import os

def parquet_to_csv(parquet_file):
    """
    Convert a parquet file to csv.
    
    Args:
        parquet_file (str): path to parquet file
    Returns:
        str: output csv file path
    """
    
    # 读取 parquet
    df = pd.read_parquet(parquet_file)

    # 生成 csv 文件名
    csv_file = os.path.splitext(parquet_file)[0] + ".csv"

    # 写入 csv
    df.to_csv(csv_file, index=False, escapechar='\\')

    return csv_file

if __name__ == "__main__":
    for root,_,files in os.walk("./dataset"):
        for file in files:
            if file.endswith(".parquet"):
                parquet_to_csv(os.path.join(root,file))
