# 医疗报告 R1 SFT 项目

基于 DeepSeek-R1 的医疗报告生成模型监督微调项目。

## 快速开始

### 环境配置
```bash
conda create -n r1_sft python=3.11
conda activate r1_sft
pip install torch transformers datasets peft accelerate bitsandbytes
pip install evaluate rouge-score nltk pandas numpy
