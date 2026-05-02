import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from SoraKu.utils.logger import setup_logger

logger = setup_logger(name="SFT Datasets to Token", component_name="Datasets to Token")

def ds_convert_hf(tokenizer, input_files):
    # 指定 Padding Token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("读取并转换数据格式")
    formatted_data = []
    
    for file_path in input_files:
        if not os.path.exits(file_path):
            continue
        
        logger.info(f"正在处理文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                raw_item = json.loads(line)
                
                # 转换格式
                messages_en = []
                messages_zh = []
                messages_jp = []
                messages_kr = []
                messages_gz = [] # gz 指代粤语
                messages_ru = []
                
                for msg in raw_item["conversations"]:
                    role = "user" if msg["from"] == "hunman" else "assistant"
                    
                    # 提取对话
                    if "value" in msg:
                        messages_en.append({"role": role, "content": msg["value"]})
                        
                    if "value_zh" in msg:
                        messages_zh.append({"role": role, "content": msg["value_zh"]})
                        
                    if "value_jp" in msg:
                        messages_jp.append({"role": role, "content": msg["value_jp"]})
                        
                    if "value_kr" in msg:
                        messages_kr.append({"role": role, "content": msg["value_kr"]})
                        
                    if "value_gz" in msg:
                        messages_gz.append({"role": role, "content": msg["value_gz"]})
                        
                    if "value_ru" in msg:
                        messages_ru.append({"role": role, "content": msg["value_ru"]})
                        
                # 将数据级划分为独立数据，分别加入训练集
                if messages_en:
                    formatted_data.append({"messages": messages_en})
                    
                if messages_zh:
                    formatted_data.append({"messages": messages_zh})
                    
                if messages_jp:
                    formatted_data.append({"messages": messages_jp})
                    
                if messages_kr:
                    formatted_data.append({"messages": messages_kr})
                    
                if messages_gz:
                    formatted_data.append({"messages": messages_gz})
                    
                if messages_ru:
                    formatted_data.append({"messages": messages_ru})
    
    # 转为 Hugging Face Dataset
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"成功加载并转换 {len(dataset)} 条数据")
    
    return dataset
            
def tokenize_and_format(tokenizer, example):
    """
    将对话套入 各个模型的 ChatML 对话模板，并转换 Token IDs。
    这里先将 labels 先复制为 input_ids
    """
    
    # 应用模板，并对句子进行 Token 化。
    
    tokenized = tokenizer.apply_chat_template(
        example["messages"],
        tokenizer=True,
        add_generation_prompt=False, # 训练数据不需要生成引导符
        return_dict=True
    )
    
    # 复制一份 labels 作为 input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集Token化")
    parser.add_argument("--tokenizer", required=True, help="分词器路径")
    parser.add_argument("--inputs", nargs='+', required=True, help="输入的多个 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="最终输出的完整体 JSONL 文件路径")
    parser.add_argument("--workers", type=int, default=8, help="并行使用的 CPU 核心数") 
    args = parser.parse_args()
    
    logger.info("加载分词器")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)
    
    dataset = ds_convert_hf(tokenizer, args.inputs)
    tokenized_dataset = dataset.map(
        tokenize_and_format(tokenizer),
        batched=True,
        remove_columns=["messages"],
        num_proc=args.workers
    )
    
    logger.info(f"保存 SFT 数据至 {args.output}")
    tokenized_dataset.save_to_disk(args.output)
    logger.info("SFT 数据集准备完毕！")
    
    # 检查处理后的数据长度
    sample_ids = tokenized_dataset[0]["input_ids"]
    logger.info(f"第一条数据 Token 长度: {len(sample_ids)}")
    print(f"数据还原文本：\n{tokenizer.decode(sample_ids)}")

