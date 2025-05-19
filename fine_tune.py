import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from evaluate import load
import torch

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 加载数据集
dataset = load_dataset("conll2003")

# 3. 定义标签（确保与CoNLL-2003完全一致）
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# 4. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 5. 改进的预处理函数（关键修复）
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt"
    )
    
    labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids_aligned = []
        
        for word_idx in word_ids:
            # 特殊token设为-100
            if word_idx is None:
                label_ids_aligned.append(-100)
            # 新单词的第一个token
            elif word_idx != previous_word_idx:
                label_ids_aligned.append(label_ids[word_idx])
            # 同一单词的后续token
            else:
                # 确保I-标签与B-标签匹配
                current_label = label_ids[word_idx]
                if current_label % 2 == 1:  # 如果是B-标签
                    label_ids_aligned.append(current_label + 1)  # 转为I-标签
                else:
                    label_ids_aligned.append(current_label)
            previous_word_idx = word_idx
        
        # 填充到max_length（确保填充值为-100）
        padding_length = 128 - len(label_ids_aligned)
        label_ids_aligned = label_ids_aligned + [-100] * padding_length
        
        # 截断并验证标签范围
        label_ids_aligned = label_ids_aligned[:128]
        assert all(l == -100 or 0 <= l < len(label_list) for l in label_ids_aligned), "Invalid label detected!"
        
        labels.append(label_ids_aligned)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 6. 应用预处理
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 7. 加载模型（确保标签数量正确）
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

# 8. 加载评估指标
metric = load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 9. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # 减小batch size以防内存不足
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),  # 启用混合精度训练
)

# 10. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 11. 开始训练（添加错误处理）
try:
    print("Starting training...")
    trainer.train()
except Exception as e:
    print(f"Training failed: {str(e)}")
    # 打印前几个样本的标签用于调试
    print("\nSample labels for debugging:")
    for i in range(3):
        print(f"Sample {i}: {tokenized_datasets['train'][i]['labels']}")
    raise

# 12. 保存模型
model.save_pretrained("./bert_finetuned")
tokenizer.save_pretrained("./bert_finetuned")
print("Training completed and model saved!")