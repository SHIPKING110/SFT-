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
