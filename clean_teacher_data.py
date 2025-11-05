import os
import json

INPUT_DIR = "./r1_generated"
OUTPUT_FILE = "./dataset/teacher_filtered.jsonl"

def clean_text(t):
    if not isinstance(t, str):
        return ""
    return t.strip().replace("\r", " ").replace("\n", " ")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total, saved = 0, 0
    cleaned_data = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".jsonl"):
            continue
        
        path = os.path.join(INPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                try:
                    item = json.loads(line)
                except:
                    continue

                q = item.get("question", "")
                r = item.get("reasoning", "")
                a = item.get("answer", "")

                # 必须三个字段都存在才保留
                if not q or not r or not a:
                    continue

                # reasoning 必须是 list，否则跳过
                if not isinstance(r, list) or len(r) == 0:
                    continue

                # 清洗
                cleaned_item = {
                    "question": clean_text(q),
                    "reasoning": [clean_text(step) for step in r],
                    "answer": clean_text(a)
                }

                cleaned_data.append(cleaned_item)
                saved += 1

    if saved == 0:
        print("⚠️ 没有任何数据通过过滤，请检查 r1_generated 格式")
        return

    # 写入 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 输出统计报告
    print("✅ 教师数据清洗完成")
    print(f"📍 输入样本数: {total}")
    print(f"📍 输出有效样本数: {saved}")
    print(f"💾 输出文件: {OUTPUT_FILE}")
    print("🔍 样例预览：")
    print(json.dumps(cleaned_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
