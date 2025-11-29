import json
import os
import sys
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm

from misc.metrics import compute_bleu, compute_rouge


def convert_to_sft(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    sft_data = []
    for d in data:
        prompt = f"Question: {d['question']}\nContext: {d['context']}"
        output = d["annotation"]
        sft_data.append({"instruction": prompt, "output": output})
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存为{output_path}")

def finetune_sft(sft_path, model_name, output_dir, epochs=2, batch_size=2):
    with open(sft_path, encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    train_dataset = Dataset.from_list(lines)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    def preprocess(example):
        prompt = example["instruction"]
        output = example["output"]
        text = prompt + "\nAnswer: " + output
        tokenized = tokenizer(text, truncation=True, max_length=512, padding='max_length')
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    train_dataset = train_dataset.map(preprocess)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"微调模型已保存到{output_dir}")

def generate_answer(model, tokenizer, question, context, max_new_tokens=64):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).split("Answer:")[-1].strip()

def evaluate(model_path, data_path):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bleu_scores = []
    rouge_l_scores = []
    for d in tqdm(data):
        pred = generate_answer(model, tokenizer, d["question"], d["context"])
        label = d["annotation"]
        # BLEU
        bleu = compute_bleu(label, pred)
        bleu_scores.append(bleu)
        # ROUGE-L
        rouge_dict = compute_rouge(label, pred)
        rouge_l = rouge_dict.get("rouge-l", {}).get("f", 0.0)
        rouge_l_scores.append(rouge_l)
    print(f"{model_path} BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"{model_path} ROUGE-L: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")


if __name__ == "__main__":
    # 路径配置
    val_path = "validation.json"
    ann_path = "all_32B.json"
    sft_path = "sft_train.jsonl"
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    sft_model_dir = "./qwen-sft"
    # 1. 转换数据
    if not os.path.exists(sft_path):
        convert_to_sft(ann_path, sft_path)
    # 2. 微调
    if not os.path.exists(sft_model_dir):
        finetune_sft(sft_path, base_model, sft_model_dir)
    # 3. 评测
    # print("评测原始模型: ")
    # evaluate(base_model, val_path)
    print("评测微调模型: ")
    evaluate(sft_model_dir, val_path)
