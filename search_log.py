import re
import sys

def search_log(log_path, keywords, context_lines=3):
    """
    在日志文件中搜索关键词，打印匹配行及上下文。
    - log_path: 日志文件路径
    - keywords: 关键词列表
    - context_lines: 显示匹配行上下各多少行
    """
    print(f"🔎 正在搜索关键词: {keywords}\n")
    matched_count = 0

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # 忽略大小写匹配
        if any(kw.lower() in line.lower() for kw in keywords):
            matched_count += 1
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)

            print(f"\n{'='*80}")
            print(f"✅ 匹配 #{matched_count} | 行号 {i+1} | 关键词命中: {line.strip()}")
            print(f"{'='*80}")
            for j in range(start, end):
                prefix = ">>> " if j == i else "    "
                print(f"{prefix}{j+1:6d}: {lines[j].rstrip()}")
            print(f"{'='*80}")

    if matched_count == 0:
        print("❌ 未找到任何匹配行，请检查关键词是否正确。")
    else:
        print(f"\n📊 共找到 {matched_count} 处匹配。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python search_log.py <日志文件路径> <关键词1> [关键词2] ...")
        print("示例: python search_log.py 20260408_18.log.txt SSW_0x00000070 Error")
        sys.exit(1)

    log_file = sys.argv[1]
    search_keywords = sys.argv[2:]
    search_log(log_file, search_keywords)