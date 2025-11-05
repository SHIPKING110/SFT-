# 医疗报告 R1 SFT 项目

基于 DeepSeek-R1 的医疗报告生成模型监督微调项目。

## 项目概述

本项目旨在通过监督微调（SFT）技术，基于 DeepSeek-R1 大语言模型构建专业的医疗报告生成系统。通过对医学问答数据进行针对性训练，提升模型在医疗领域的专业性和准确性。

## 环境配置

```bash
# 创建 conda 环境
conda create -n r1_sft python=3.11 -y
conda activate r1_sft

# 安装核心依赖
pip install torch==2.5.1 transformers==4.57.1 datasets peft accelerate bitsandbytes

# 安装评估和工具依赖
pip install evaluate rouge-score nltk pandas numpy
```

## 快速开始
```bash
# 1. 数据准备与预处理
python data/generated_converted_data.py
python data/generated_stf_r1_data.py  
python data/split_stf_r1_data.py

# 2. 下载基础模型
python download_model/download_model.py

# 3. 模型训练
python scripts/train_stf_r1_train_val.py

# 4. 模型评估
python scripts/evaluate_model.py

# 5. 模型测试与对话
python scripts/chat_with_model.py
```

## 数据格式
```bash
{
  "input": "患者：医生，我最近出现持续性干咳、胸痛和呼吸困难，可能是什么原因？",
  "output": "根据您描述的症状（持续性干咳、胸痛、呼吸困难），可能的原因包括：\n\n1. 呼吸系统疾病：\n   - 支气管炎或肺炎\n   - 哮喘或COPD急性发作\n   - 胸膜炎\n\n2. 心血管疾病：\n   - 心功能不全\n   - 心包炎\n\n3. 其他原因：\n   - 胃食管反流\n   - 焦虑症引发的躯体症状\n\n建议尽快就医进行详细检查，包括胸部X光、肺功能测试等。"
}
```

## 目录结构
```bash
r1_sft/
├── conversations/               # 存放模型会话（自动生成）
├── dataset/                     # 清洗后可用的数据集（下一步会用）
│      ├──sft_r1_train.jsonl     # 医疗训练数据：训练集（自动生成）
│      ├──sft_r1_val.jsonl       # 医疗训练数据：验证集（自动生成）
│      ├──sft_r1_data.jsonl      # 医疗训练数据：converted_data.jsonl转换（自动生成）
│      ├──train.jsonl            # 训练集（自动生成）
│      ├──val.jsonl              # 验证集（自动生成）
│      └──sft_train.jsonl        # SFT数据（自动生成）
├── download_model/          
│        └──download_model.py    # 下载模型脚本
├── model/                       # 存放模型与训练权重
│       ├──deepseek_ai/           # 下载的Deepseek预训练模型
│       └──deepseek_r1_1.5b_lora/ # 模型训练权重
│                 ├──best_model             # 最优模型权重（自动生成）
│                 ├──checkpoint-480         # 训练保存节点1（自动生成）
│                 ├──checkpoint-720         # 训练保存节点2（自动生成）
│                 ├──checkpoint             # 训练保存节点（自动生成）
│                 └──training_logs          # 训练日志
├── modelscope_r1_data/          # 存放魔搭开源数据
│        ├──r1_data_example.jsonl# 魔搭社区医疗开源原始数据
│        └──converted_data.jsonl # r1_data_example.jsonl转换后数据（自动生成）
├── r1_generated/                # 用R1生成的推理答案
│        ├──teacher_filtered.jsonl# 001.jsonl、002.jsonl、003.jsonl合并后数据（自动生成）
│        ├──001.jsonl            # R1生成数据示例
│        ├──002.jsonl
│        └──003.jsonl
├── scripts/ 
│        ├── train_stf_r1_train_val.py   # 训练脚本——优化版      
│        ├── train_distill.py            # 训练脚本
│        ├── evaluate_model.py           # 评估脚本
│        ├── chat_with_model.py          # 训练后模型多轮对话脚本
│        └── compare/
│              ├── evaluation_results/   # 评估结果输出目录
│              ├── compare_str_r1.py     # 评估脚本
│              └── install_deps.py       # 评估依赖安装脚本
├── build_sft_dataset.py          # SFT数据转换程序：转换dataset/sft_train.jsonl
├── clean_teacher_data.py         # 
├── compare.py                    # 评估测试模型权重
├── generated_converted_data.py   # 转换modelscope_r1_data/r1_data_example.jsonl
├── generated_stf_r1_data.py      # 转换converted_data.jsonl为SFT训练格式
├── generated_teacher_filtered.py # 合并001.jsonl、002.jsonl、003.jsonl数据
├── split_sft_train.py            # 划分sft_train为train.jsonl和val.jsonl 
├── split_sft_r1_data.py          # 划分sft_r1_data.jsonl→（sft_r1_train.jsonl与sft_r1_val.jsonl）
└── test_model.py                 # 测试下载模型基本对话能力
                      # 
```
