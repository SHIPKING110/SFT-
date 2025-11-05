# scripts/train_distill.py
import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------- CONFIG --------------
MODEL_NAME = "/workspace/AI_funning/r1_distill_finance/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./models/deepseek_r1_1.5b_lora"
TRAIN_FILE = "./dataset/sft_r1_data.jsonl"
VAL_FILE = "./dataset/sft_r1_val.jsonl"

# Hardware / train config - 针对16GB显存优化
MICRO_BATCH_SIZE = 2        # 增加到2，因为数据量不大
GRAD_ACCUMULATION_STEPS = 4 # 有效batch = 2 * 4 = 8
EPOCHS = 100                # 增加到5个epoch
LEARNING_RATE = 1e-4        # 降低学习率
WEIGHT_DECAY = 0.01         # 增加权重衰减防止过拟合
WARMUP_STEPS = 100          # 增加warmup
SAVE_STRATEGY = "steps"
LOGGING_STEPS = 20
SAVE_STEPS = 100            # 每100步保存一次checkpoint
SAVE_TOTAL_LIMIT = 3        # 只保留3个最佳checkpoint

USE_4BIT = True
USE_TEACHER_LOGITS = False
LAMBDA_KL = 0.5

# Checkpoint配置
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
# ------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 智能保存回调 ----------------
class SmartSaveCallback(TrainerCallback):
    def __init__(self, output_dir, save_steps=100, save_total_limit=3):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.best_eval_loss = float('inf')
        self.checkpoint_history = []
        
        # 创建目录
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 每save_steps步保存一次checkpoint
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            
            # 保存训练状态
            torch.save({
                'global_step': state.global_step,
                'epoch': state.epoch,
                'train_loss': state.log_history[-1]['loss'] if state.log_history else None,
            }, os.path.join(checkpoint_dir, "training_state.pt"))
            
            print(f"✅ 检查点已保存: {checkpoint_dir}")
            
            # 管理checkpoint数量
            self.manage_checkpoints()
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']
            
            # 如果当前loss更好，保存最佳模型
            if current_loss < self.best_eval_loss:
                self.best_eval_loss = current_loss
                
                # 保存最佳模型
                if os.path.exists(BEST_MODEL_DIR):
                    import shutil
                    shutil.rmtree(BEST_MODEL_DIR)
                
                kwargs['model'].save_pretrained(BEST_MODEL_DIR)
                kwargs['tokenizer'].save_pretrained(BEST_MODEL_DIR)
                
                # 保存评估信息
                eval_info = {
                    'eval_loss': current_loss,
                    'global_step': state.global_step,
                    'epoch': state.epoch
                }
                with open(os.path.join(BEST_MODEL_DIR, "eval_info.json"), 'w') as f:
                    json.dump(eval_info, f, indent=2)
                
                print(f"🎯 新的最佳模型已保存! eval_loss: {current_loss:.4f}")
    
    def manage_checkpoints(self):
        """管理checkpoint数量，只保留最好的几个"""
        checkpoints = []
        for item in os.listdir(CHECKPOINT_DIR):
            checkpoint_path = os.path.join(CHECKPOINT_DIR, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[1])
                    state_file = os.path.join(checkpoint_path, "training_state.pt")
                    if os.path.exists(state_file):
                        state = torch.load(state_file)
                        checkpoints.append((step, state['train_loss'] if state['train_loss'] else float('inf'), checkpoint_path))
                except:
                    continue
        
        # 按loss排序，保留最好的
        checkpoints.sort(key=lambda x: x[1])
        
        # 删除多余的checkpoint
        if len(checkpoints) > self.save_total_limit:
            for step, loss, path in checkpoints[self.save_total_limit:]:
                import shutil
                shutil.rmtree(path)
                print(f"🗑️  删除检查点: {path}")

# ---------------- util ----------------
def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def find_latest_checkpoint():
    """查找最新的checkpoint"""
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoints = []
    for item in os.listdir(CHECKPOINT_DIR):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, item)
        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, checkpoint_path))
            except:
                continue
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]
    return None

# ---------------- dataset loading & tokenization ----------------
def preprocess(dataset_items, tokenizer, max_length=1024):
    inputs = []
    for it in dataset_items:
        inp = it["input"]
        out = it["output"]
        text = inp + "\n" + out
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        input_ids = tok["input_ids"]
        inputs.append({"input_ids": input_ids, "attention_mask": tok["attention_mask"]})
    return inputs

# ---------------- collator ----------------
@dataclass
class DataCollatorForCausal:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# ---------------- main ----------------
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model -- this may take a while")
    
    # 检查是否有之前的checkpoint
    resume_from_checkpoint = find_latest_checkpoint()
    if resume_from_checkpoint:
        print(f"🎯 发现检查点，从 {resume_from_checkpoint} 恢复训练")
    
    # load with 4bit if requested
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME if not resume_from_checkpoint else resume_from_checkpoint,
            quantization_config=bnb_config, 
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME if not resume_from_checkpoint else resume_from_checkpoint,
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    # apply LoRA via PEFT
    lora_config = LoraConfig(
        r=16,  # 增加rank以获得更好效果
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load data
    train_items = read_jsonl(TRAIN_FILE)
    val_items = read_jsonl(VAL_FILE)
    
    print(f"训练数据: {len(train_items)} 条")
    print(f"验证数据: {len(val_items)} 条")
    
    train_tok = preprocess(train_items, tokenizer)
    val_tok = preprocess(val_items, tokenizer)

    collator = DataCollatorForCausal(tokenizer=tokenizer)

    # 计算总步数
    total_steps = (len(train_tok) * EPOCHS) // (MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
    print(f"总训练步数: {total_steps}")
    
    # 计算保存步数 - 每100步或每个epoch保存一次
    save_steps = max(100, len(train_tok) // (MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
    print(f"保存步数: {save_steps}")

    # training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        eval_strategy="steps",  # 改为按步数评估
        eval_steps=save_steps,  # 与保存步数一致
        save_strategy="steps",
        save_steps=save_steps,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,
        save_total_limit=1,  # 让callback管理checkpoint
        dataloader_pin_memory=True,
        report_to="none",
        load_best_model_at_end=False,  # 让callback管理最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # 初始化callback
    smart_save_callback = SmartSaveCallback(OUTPUT_DIR, save_steps=save_steps, save_total_limit=SAVE_TOTAL_LIMIT)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[smart_save_callback]
    )

    # 训练
    print("开始训练...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存最终模型
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print("🎉 训练完成!")
    print(f"最终模型保存至: {final_model_dir}")
    print(f"最佳模型保存至: {BEST_MODEL_DIR}")

if __name__ == "__main__":
    main()