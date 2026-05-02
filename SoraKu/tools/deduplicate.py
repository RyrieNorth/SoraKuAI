import json
import argparse
import os
import jieba
from datasketch import MinHash, MinHashLSH, LeanMinHash
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

from SoraKu.utils.logger import setup_logger
logger = setup_logger(name="Deduplicate", component_name="Deduplicate")

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def chunk_reader(iterable, chunk_size):
    """生成器：按指定大小分块读取文件，防止内存爆满"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

def process_batch(lines):
    """
    批处理计算函数（在子进程中运行）
    一次性接收 N 行，返回解析后的数据和紧凑版 MinHash 列表
    """
    results = []
    for line in lines:
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            text = data.get("text", "") 
            if not text:
                continue
                
            m = MinHash(num_perm=128)
            # jieba.lcut 返回列表，在短文本下效率稍高
            words = jieba.lcut(text)
            for word in words:
                if len(word.strip()) > 0:
                    m.update(word.encode('utf8'))
            
            # 使用 LeanMinHash 极大减少跨进程传输和 LSH 存储的内存占用
            lean_m = LeanMinHash(m)
            results.append((data, lean_m))
        except json.JSONDecodeError:
            continue
            
    return results

def parallel_deduplicate(input_files, output_file, threshold=0.85, max_workers=8, batch_size=2000):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    total_processed = 0
    total_saved = 0
    total_duplicates = 0
    
    logger.info("开始去重与合并...")
    logger.info(f"Workers: {max_workers} | Batch Size: {batch_size}")
    logger.info(f"相似度阈值: {threshold * 100}%")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for file_path in input_files:
                if not os.path.exists(file_path):
                    continue
                    
                print(f"正在并行处理文件: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as fin:
                    # 使用 executor.map 按块流水线执行
                    # 这样内存中最多只保留 max_workers * batch_size 条数据
                    for batch_results in executor.map(process_batch, chunk_reader(fin, batch_size)):
                        
                        # 收集批次结果并在主线程中写入 LSH
                        for data, lean_m in batch_results:
                            total_processed += 1
                            
                            result = lsh.query(lean_m)
                            
                            if len(result) > 0:
                                total_duplicates += 1
                            else:
                                lsh.insert(str(total_processed), lean_m)
                                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                                total_saved += 1
                                
                            if total_processed % 50000 == 0:
                                logger.info(f"扫描: {total_processed} 条 | 拦截: {total_duplicates} 条 | 保留: {total_saved} 条")

    logger.info(f"总计处理: {total_processed} 条")
    logger.info(f"最终保留: {total_saved} 条")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多文件 LSH 文本去重脚本")
    parser.add_argument("--inputs", nargs='+', required=True, help="输入的多个 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="最终输出的完整体 JSONL 文件路径")
    parser.add_argument("--threshold", type=float, default=0.85, help="相似度阈值")
    parser.add_argument("--workers", type=int, default=8, help="并行使用的 CPU 核心数") 
    parser.add_argument("--batch_size", type=int, default=2000, help="每个子进程每次领取的任务行数") 
    
    args = parser.parse_args()
    parallel_deduplicate(args.inputs, args.output, args.threshold, args.workers, args.batch_size)