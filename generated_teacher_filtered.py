import json
import glob

input_files = glob.glob("r1_generated/*.jsonl")
output_file = "./dataset/teacher_filtered.jsonl"

def check_and_standardize(record):
    # 检查 question
    if not record.get("question"):
        return None
    # 检查 reasoning
    reasoning = record.get("reasoning", [])
    if not isinstance(reasoning, list) or len(reasoning) < 3:
        return None
    # 检查 answer
    if not record.get("answer"):
        return None
    # 标准化：去掉多余空格
    record["question"] = record["question"].strip()
    record["reasoning"] = [step.strip() for step in reasoning]
    record["answer"] = record["answer"].strip()
    return record

with open(output_file, "w", encoding="utf-8") as fout:
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    std_record = check_and_standardize(record)
                    if std_record:
                        fout.write(json.dumps(std_record, ensure_ascii=False) + "\n")
                except:
                    continue  # 跳过无法解析的行
print("标准化完成，输出文件:", output_file)
