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
    default_data_collator
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------- CONFIG --------------
MODEL_NAME = "/workspace/AI_funning/r1_distill_finance/models/deepseek-ai/deepseek-coder-1.3b-instruct"   # <-- 替换成你要微调的 base model
OUTPUT_DIR = "./models/distilled_1.3b_lora"
TRAIN_FILE = "./dataset/train.jsonl"
VAL_FILE = "./dataset/val.jsonl"

# Hardware / train config
MICRO_BATCH_SIZE = 3        # per step per gpu
GRAD_ACCUMULATION_STEPS = 8 # effective batch = MICRO_BATCH_SIZE * grad_accum
EPOCHS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3

USE_4BIT = True
USE_TEACHER_LOGITS = False    # 如果你有教师 logits（同长度 token logits），把 True
LAMBDA_KL = 0.5              # 仅当 USE_TEACHER_LOGITS=True 有效
# ------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# ---------------- dataset loading & tokenization ----------------
def preprocess(dataset_items, tokenizer, max_length=1024):
    inputs = []
    for it in dataset_items:
        inp = it["input"]
        out = it["output"]
        text = inp + "\n" + out  # model sees input + output
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        input_ids = tok["input_ids"]
        # labels: we want model to predict output tokens, but we'll let causal LM predict whole sequence.
        # For causal LM SFT, labels = input_ids (teacher forcing). But to avoid teacher forcing on input,
        # we can set label -100 for input part so loss computed only on output or compute on whole sequence.
        # Here: compute loss on entire sequence so model learns to generate reasoning+answer conditioned on prompt.
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
    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model -- this may take a while")
    # load with 4bit if requested
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
        # prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    # apply LoRA via PEFT
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],  # 视模型架构调整：例如 ['q_proj','k_proj','v_proj','o_proj']
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load data
    train_items = read_jsonl(TRAIN_FILE)
    val_items = read_jsonl(VAL_FILE)
    train_tok = preprocess(train_items, tokenizer)
    val_tok = preprocess(val_items, tokenizer)

    collator = DataCollatorForCausal(tokenizer=tokenizer)

    # training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy=SAVE_STRATEGY,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_pin_memory=True,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator
    )

    # If you have teacher logits and want KL distillation, you'll need a custom training loop.
    # Here we run standard Trainer (CE loss on labels).
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training complete. Model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
