Medical Report R1 SFT Project
基于DeepSeek-R1的医疗报告生成模型监督微调项目。

🚀 快速开始
🔧 环境配置
bash
conda create -n r1_sft python=3.11
conda activate r1_sft
pip install torch transformers datasets peft accelerate bitsandbytes
pip install evaluate rouge-score nltk pandas numpy
📊 数据准备
bash
python generated_converted_data.py
python generated_stf_r1_data.py  
python split_sft_r1_data.py
🤖 模型下载
bash
python download_model/download_model.py
🎯 模型训练
bash
python scripts/train_stf_r1_train_val.py
📈 模型评估
bash
python scripts/evaluate_model.py
💬 对话测试
bash
python scripts/chat_with_model.py
✨ 项目特点
监督微调医疗问答模型

推理链生成结构化输出

LoRA高效微调（仅1.02%参数）

智能保存最佳模型

完整评估体系

📝 数据格式
json
{
  "input": "用户：肝硬化晚期有哪些临床表现？",
  "output": "<reasoning>\n1. 肝硬化晚期主要表现包括...\n2. 肝功能损害导致...\n</reasoning>\n答：肝硬化晚期临床表现主要包括..."
}
📄 许可证
MIT License
