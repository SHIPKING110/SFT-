import json
import random

INPUT_FILE = "./dataset/sft_train.jsonl"
TRAIN_FILE = "./dataset/train.jsonl"
VAL_FILE = "./dataset/val.jsonl"
VAL_RATIO = 0.1  # 可调

def main():
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    random.shuffle(data)
    total = len(data)
    val_size = max(1, int(total * VAL_RATIO))

    val_data = data[:val_size]
    train_data = data[val_size:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ 数据划分完成")
    print(f"📊 总样本: {total}")
    print(f"📘 训练集: {len(train_data)} -> {TRAIN_FILE}")
    print(f"🧪 验证集: {len(val_data)} -> {VAL_FILE}")

    print("\n🔍 验证集示例：")
    print(json.dumps(val_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
