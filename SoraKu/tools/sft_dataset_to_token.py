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
    
    # 语言字段映射
    lang_keys = ["value", "value_zh", "value_jp", "value_kr", "value_gz", "value_ru"]
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在，已跳过: {file_path}")
            continue
        
        logger.info(f"正在处理文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                raw_item = json.loads(line)
                
                # 使用字典推导式为所有支持的语种初始化空的 message 列表
                lang_messages = {key: [] for key in lang_keys}
                
                for msg in raw_item["conversations"]:
                    role = "user" if msg["from"] == "human" else "assistant"
                    
                    # 遍历所有定义的语言字段，如果存在则将其加入对应的列表中
                    for key in lang_keys:
                        if key in msg and msg[key]: # 确保存在且不为空
                            lang_messages[key].append({"role": role, "content": msg[key]})
                
                # 将收集到的各语种对话加入最终的训练数据列表中
                for key in lang_keys:
                    if lang_messages[key]:
                        formatted_data.append({"messages": lang_messages[key]})
    
    # 转为 Hugging Face Dataset
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"成功加载并转换 {len(dataset)} 条数据")
    
    return dataset
            
def tokenize_and_format(examples, tokenizer):
    """
    将对话套入 各个模型的 ChatML 对话模板，并转换 Token IDs。
    这里先将 labels 先复制为 input_ids
    """
    
    # 应用模板，并对句子进行 Token 化。
    tokenized = tokenizer.apply_chat_template(
        examples["messages"],
        tokenize=True,
        add_generation_prompt=False, # 训练数据不需要生成引导符
        return_dict=True             # 返回包含 input_ids 等键的字典
    )
    
    # 复制一份 labels 作为 input_ids
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]] 
    return tokenized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 数据集 Token 化",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="分词器路径")
    parser.add_argument("--inputs", nargs='+', required=True, help="输入的多个 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="最终输出的完整体 JSONL 文件路径")
    parser.add_argument("--workers", type=int, default=8, help="并行使用的 CPU 核心数") 
    args = parser.parse_args()
    
    logger.info("加载分词器中...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=True)
    logger.info("加载完毕！")
    
    dataset = ds_convert_hf(tokenizer, args.inputs)
    tokenized_dataset = dataset.map(
        tokenize_and_format,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["messages"],
        num_proc=args.workers
    )
    
    logger.info(f"保存 SFT 数据至 {args.output}")
    tokenized_dataset.save_to_disk(args.output)
    logger.info("SFT 数据集准备完毕！")
    
    # 检查处理后的数据长度
    if len(tokenized_dataset) > 0:
        sample_ids = tokenized_dataset[0]["input_ids"]
        logger.info(f"第一条数据 Token 长度: {len(sample_ids)}")
        print(f"数据还原文本：\n{tokenizer.decode(sample_ids)}")
    else:
        logger.warning("转换后的数据集为空，请检查输入文件格式。")
