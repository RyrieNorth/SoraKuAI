import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from SoraKu.utils.logger import setup_logger

logger = setup_logger(name="SFT Datasets to Token", component_name="SFT Datasets to Token")


def convert_datasets(input_file, val_percent=0.01, seed=42):
    logger.info(f"加载 JSONL 数据集：{input_file} ...")
    raw_datasets = load_dataset("json", data_files={"train": input_file})
    
    # 动态切分验证集
    logger.info(f"正在切分验证集 ({val_percent * 100}%)")
    
    split_datasets = raw_datasets["train"].train_test_split(
        test_size=val_percent,
        seed=seed
    )
    
    logger.info(f"训练集数量：{len(split_datasets['train'])} 条")
    logger.info(f"验证集数量：{len(split_datasets['test'])} 条")
    
    return split_datasets


def tokenize_and_format(tokenizer, examples):
    """
    步骤 1: 将纯文本全部转换为 Token IDs。
    预训练不需要截断或填充。
    """
    eos_token_id = tokenizer.eos_token_id
    
    outputs = tokenizer(examples["text"], truncation=False, padding=False)
    
    # 在每篇文章的末尾加上 EOS (End of Sentence) Token
    for input_ids in outputs["input_ids"]:
        input_ids.append(eos_token_id)
    
    return {"input_ids": outputs["input_ids"]}


def group_texts(examples, block_size=2048):
    """
    步骤 2: 将所有短序列拼成超长序列，再按 BLOCK_SIZE 均匀切割。
    """
    
    # 1.将这一批次里所有的 input_ids 拼接为一个超大一维数组
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    
    # 2.计算拼接后的长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 3.若长度不够 BLOCK_SIZE，则直接丢弃（防止维度对不齐）
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # 4.按 BLOICK_SIZE 切块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # 5.复制一份 labels 作为 input_ids
    result["labels"] = result["input_ids"].copy()
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集Token化")
    parser.add_argument("--tokenizer", required=True, help="分词器路径")
    parser.add_argument("--input", required=True, help="输入的多个 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="最终输出的完整体 JSONL 文件路径")
    parser.add_argument("--workers", type=int, default=8, help="并行使用的 CPU 核心数") 
    parser.add_argument("--block_size", type=int, default=2048, help="每条数据集的长度")
    parser.add_argument("--val_percent", type=float, default=0.01, help="切分 1% 作为验证集")
    parser.add_argument("--seed", type=int, default=42, help="切分数据集固定种子")
    args = parser.parse_args()
    
    logger.info("加载分词器")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)
    
    split_datasets = convert_datasets(args.inputs, args.val_percent, args.seed)
    
    logger.info("正在进行 Token 处理")
    