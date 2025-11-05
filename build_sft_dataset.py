import json

INPUT_FILE = "./dataset/teacher_filtered.jsonl"
OUTPUT_FILE = "./dataset/sft_train.jsonl"

def convert_to_sft(item):
    question = item["question"]
    reasoning = item["reasoning"]
    answer = item["answer"]

    # 格式化 reasoning 为序号列表
    reasoning_text = "<reasoning>\n"
    for i, step in enumerate(reasoning, start=1):
        reasoning_text += f"{i}. {step}\n"
    reasoning_text += "</reasoning>"

    output_text = f"{reasoning_text}\n答：{answer}"

    return {
        "input": f"用户：{question}",
        "output": output_text
    }

def main():
    count = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            sft_item = convert_to_sft(item)
            fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            count += 1

    print("✅ SFT数据构建完成！")
    print(f"📍 输入数据: {INPUT_FILE}")
    print(f"📍 输出数据: {OUTPUT_FILE}")
    print(f"📊 样本数: {count}")

    # 示例展示
    print("\n🔍 示例样本：")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        print(f.readline())

if __name__ == "__main__":
    main()
