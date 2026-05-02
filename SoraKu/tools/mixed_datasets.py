import argparse
from datasets import load_from_disk, concatenate_datasets
from SoraKu.utils.logger import setup_logger

logger = setup_logger(name="Mixed Dataset", component_name="Mixed Dataset")

logger.info("加载已处理的数据集...")
identity = load_from_disk("./datasets/sft_v1/identity.jsonl")

def mixed_dataset(input_files, output_file):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混合数据集")
    parser.add_argument("--inputs", nargs='+', required=True, help="输入的多个 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="最终输出的完整体 JSONL 文件路径")
    
    args = parser.parse_args()