**Medical Report R1 SFT Project**
一个基于DeepSeek-R1的医疗报告生成模型监督微调项目，通过SFT技术让模型学会生成包含推理过程的专业医疗回答。

**📁 完整项目结构**
r1_sft_train/
├── conversations/               # 存放模型会话（自动生成）
├── dataset/                     # 清洗后可用的数据集
│   ├── sft_r1_train.jsonl      # 医疗训练数据：训练集（自动生成）
│   ├── sft_r1_val.jsonl        # 医疗训练数据：验证集（自动生成）
│   ├── sft_r1_data.jsonl       # 医疗训练数据：converted_data.jsonl转换（自动生成）
│   ├── train.jsonl             # 训练集（自动生成）
│   ├── val.jsonl               # 验证集（自动生成）
│   └── sft_train.jsonl         # SFT数据（自动生成）
├── download_model/          
│   └── download_model.py       # 下载模型脚本
├── model/                       # 存放模型与训练权重
│   ├── deepseek_ai/            # 下载的Deepseek预训练模型
│   └── deepseek_r1_1.5b_lora/  # 模型训练权重
│       ├── best_model          # 最优模型权重（自动生成）
│       ├── checkpoint-480      # 训练保存节点1（自动生成）
│       ├── checkpoint-720      # 训练保存节点2（自动生成）
│       ├── checkpoint          # 训练保存节点（自动生成）
│       └── training_logs       # 训练日志
├── modelscope_r1_data/         # 存放魔搭开源数据
│   ├── r1_data_example.jsonl   # 魔搭社区医疗开源原始数据
│   └── converted_data.jsonl    # r1_data_example.jsonl转换后数据（自动生成）
├── r1_generated/               # 用R1生成的推理答案
│   ├── teacher_filtered.jsonl  # 001.jsonl、002.jsonl、003.jsonl合并后数据（自动生成）
│   ├── 001.jsonl               # R1生成数据示例
│   ├── 002.jsonl
│   └── 003.jsonl
├── scripts/ 
│   ├── train_stf_r1_train_val.py   # 训练脚本——优化版      
│   ├── train_distill.py            # 训练脚本
│   ├── evaluate_model.py           # 评估脚本
│   ├── chat_with_model.py          # 训练后模型多轮对话脚本
│   └── compare/
│       ├── evaluation_results/     # 评估结果输出目录
│       ├── compare_str_r1.py       # 评估脚本
│       └── install_deps.py         # 评估依赖安装脚本
├── build_sft_dataset.py           # SFT数据转换程序：转换dataset/sft_train.jsonl
├── clean_teacher_data.py          # 清理教师数据
├── compare.py                     # 评估测试模型权重
├── generated_converted_data.py    # 转换modelscope_r1_data/r1_data_example.jsonl
├── generated_stf_r1_data.py       # 转换converted_data.jsonl为SFT训练格式
├── generated_teacher_filtered.py  # 合并001.jsonl、002.jsonl、003.jsonl数据
├── split_sft_train.py             # 划分sft_train为train.jsonl和val.jsonl 
├── split_sft_r1_data.py           # 划分sft_r1_data.jsonl→（sft_r1_train.jsonl与sft_r1_val.jsonl）
├── test_model.py                  # 测试下载模型基本对话能力
└── README.md                      # 项目说明文档

**🚀 快速开始**
环境配置
创建conda环境：
conda create -n r1_sft python=3.11
conda activate r1_sft

**安装依赖：**
pip install torch==2.5.1 transformers==4.57.1 datasets peft accelerate bitsandbytes
pip install evaluate rouge-score nltk absl-py pandas numpy



