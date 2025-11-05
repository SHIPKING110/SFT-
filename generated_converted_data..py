import json
import os

def convert_data_format(input_file, output_dir):
    """
    将原始数据格式转换为目标格式
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录路径
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(output_dir, "converted_data.jsonl")
    
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                # 解析原始JSON数据
                original_data = json.loads(line.strip())
                
                # 构建目标格式
                target_format = {
                    "question": original_data.get("question", ""),
                    "reasoning": [original_data.get("think", "")],  # 将think作为列表的第一个元素
                    "answer": original_data.get("answer", "")
                }
                
                # 写入转换后的数据
                outfile.write(json.dumps(target_format, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}")
                continue
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue
    
    print(f"转换完成！共转换 {converted_count} 条数据")
    print(f"输出文件: {output_file}")

# 文件路径
input_file = "./raw_questions/r1_data_example.jsonl"
output_dir = "./raw_questions"

# 执行转换
if __name__ == "__main__":
    if os.path.exists(input_file):
        convert_data_format(input_file, output_dir)
    else:
        print(f"输入文件不存在: {input_file}")
        print("请检查文件路径是否正确")