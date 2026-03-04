import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
print(os.environ['HF_ENDPOINT'])
def download_dataset():
    repo_id = "cais/wmdp"
    # Use current directory path for Linux
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(current_dir, "wmdp")

    print(f"正在从镜像站下载 {repo_id} 到 {local_dir} ...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",      # cais/wmdp 是数据集，必须指定 type，否则默认按模型找会报错
            local_dir=local_dir,      # 下载到的本地目录
            local_dir_use_symlinks=False, # 设置为 False 以下载实际文件，而不是缓存的软链接
            resume_download=True,     # 支持断点续传
        )
        print("下载完成！")
    except Exception as e:
        print(f"下载出错: {e}")

if __name__ == "__main__":
    download_dataset()