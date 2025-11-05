from modelscope import snapshot_download

# 指定下载目录

#1、编码模型
# model_dir = snapshot_download(
#     'deepseek-ai/deepseek-coder-1.3b-instruct',
#     cache_dir='/workspace/AI_funning/r1_distill_finance/models'
# )

#2、病例报告生成模型
# model_dir = snapshot_download(
#     'hjc666666/MedicalReport-DeepSeek-R1-Distill-Qwen2.5-3B',
#     cache_dir='/workspace/AI_funning/r1_distill_finance/models'
# )

#3
model_dir = snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    cache_dir='./models'
)

print(f"模型下载到：{model_dir}")