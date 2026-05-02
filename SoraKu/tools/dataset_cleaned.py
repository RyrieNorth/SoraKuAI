import json
import argparse
import re
from SoraKu.utils.logger import setup_logger

logger = setup_logger(name="Dataset cleaner", component_name="Dataset cleaner")

def clean_and_convert(
    input_path,
    output_path,
    lang='zh'
):
    # 过滤关键字
    identity_patterns = [
        # 过滤角色认知
        re.compile(r"(作为一个(?:人工智能|大型?语言模型|AI助手?)|As an? (?:AI|artificial intelligence|large language model))", re.IGNORECASE),
        re.compile(r"(我是(?:一个)?(?:人工智能|大型?语言模型|虚拟助手)|I am an? (?:AI|language model|virtual assistant))", re.IGNORECASE),
        re.compile(r"(我是(?:一个)?(?:人工智能|大型?语言模型|虚拟助手)|I am an? (?:AI|language model|virtual assistant))", re.IGNORECASE),
        
        # 过滤机构关联
        re.compile(r"(我|AI|助手|程序|模型).{0,15}由\s*(OpenAI|Anthropic|Google|Meta|DeepSeek|MiniMax|智谱|阿里|百度|字节|百川|月之暗面|科大讯飞|通义实验室|豆包).{0,10}(开发|训练|创建|提供)"),
        re.compile(r"(trained|developed|created|powered)\s*by\s*(OpenAI|Anthropic|Google|Meta|DeepSeek|Mistral|Qwen|MiniMax)", re.IGNORECASE),
        
        # 过滤模式名称
        re.compile(r"(我是|I am|This is)\s*(ChatGPT|GPT-3|GPT-4|GPT-5|Claude|Gemini|Bard|文心一言|豆包|Kimi|通义千问|Qwen|MiniMax)", re.IGNORECASE),
        
        # 过滤免责声明
        re.compile(r"(截至我的知识库|对不起|知识截止于|my knowledge cutoff|I don't have personal opinions|作为一个.*?我无法|As an AI.*?I cannot)", re.IGNORECASE)
    ]
    
    # 对用户提问进行过滤
    human_identity_patterns = [
        # 过滤直接打招呼或称呼
        re.compile(r"^(你好|嘿|您好|hi|hello|hey|请问)[，,\s]*(ChatGPT|chatgpt|chatGPT|GPT-3|GPT-4|GPT-5|Claude|Gemini|Bard|DeepSeek|Kimi|文心一言|豆包|通义千问)", re.IGNORECASE),
        
        # 过滤用户给模型做角色设定
        re.compile(r"(你作为|作为|身为|你是)[一一个\s]*(ChatGPT|GPT-3|GPT-4|GPT-5|Claude|Gemini|Bard|DeepSeek|Kimi|文心一言|豆包|通义千问)", re.IGNORECASE),
        re.compile(r"(亲爱的)(ChatGPT|GPT-3|GPT-4|GPT-5|Claude|Gemini|Bard|DeepSeek|Kimi|文心一言|豆包|通义千问)", re.IGNORECASE),
        
        # 过滤直接祈使句呼叫
        re.compile(r"^(ChatGPT|GPT-3|GPT-4|Claude|Gemini|Bard|DeepSeek|Kimi|文心一言|豆包|通义千问)[，,：:\s]+(帮我|请|你能|告诉我|写)", re.IGNORECASE)
    ]
    
    processed_count = 0
    skipped_count = 0
    
    logger.info(f"开始处理文件: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:
            
            for line_num, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                
                try:
                    raw_data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败，跳过处理。")
                    skipped_count += 1
                    continue
                
                text_blocks = []
                conversation = raw_data.get("conversation", [])
                
                # 标记对话是否被污染
                conversation_is_toxic = False
                
                for turn in conversation:
                    human_text = turn.get("human", "").strip()
                    assistant_text = turn.get("assistant", "").strip()
                    
                    if human_text and assistant_text:
                        for pattern in human_identity_patterns:
                            if pattern.search(human_text):
                                conversation_is_toxic = True
                                logger.debug(f"\n命中正则：{pattern.pattern}\n原文：{human_text[:50]}...")
                                break
                            
                        if conversation_is_toxic:
                            break
                        
                        for pattern in identity_patterns:
                            if pattern.search(assistant_text):
                                conversation_is_toxic = True
                                logger.debug(f"\n命中正则：{pattern.pattern}\n原文：{assistant_text[:100]}...")
                                break
                        
                        if conversation_is_toxic:
                            break
                        
                        text_blocks.append(f"User: {human_text}\nAssistant: {assistant_text}")
                        
                if conversation_is_toxic:
                    skipped_count += 1
                    continue
                
                full_text = "\n\n".join(text_blocks)
                
                if len(full_text) < 50:
                    skipped_count += 1
                    continue
                
                conv_id = raw_data.get("conversation_id", f"unknown_{line_num}")
                
                clean_data = {
                    "id": f"dialogue_{conv_id}",
                    "text": full_text,
                    "meta": {
                        "source": "scraped_dialogue", 
                        "language": f"{lang}",
                        "domain": "conversation",
                        "length": len(full_text)
                    }
                }
                
                fout.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    logger.info(f"已处理 {processed_count} 条有效数据...")
                    
    logger.info("处理完成！")
    logger.info(f"总计成功转换: {processed_count} 条")
    logger.info(f"因含违禁身份词/过短而丢弃: {skipped_count} 条")
    logger.info(f"输出文件保存在: {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将原始对话 JSONL 转换为预训练标准纯文本 JSONL，并包含基于正则的高级清洗功能。")
    parser.add_argument("--lang", default='zh', help="数据集的语言类型 (例如 zh、en)")
    parser.add_argument("--input", required=True, help="输入的原始 JSONL 文件路径 (例如 raw.jsonl)")
    parser.add_argument("--output", required=True, help="输出的清洗后 JSONL 文件路径 (例如 clean.jsonl)")
    
    args = parser.parse_args()
    clean_and_convert(args.input, args.output, args.lang)