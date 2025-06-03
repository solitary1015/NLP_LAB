import os
import json
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
import torch

# 设置输出目录
output_dir = "conll2003_bert_processed"
os.makedirs(output_dir, exist_ok=True)

# 加载训练好的微调模型
device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline(
    "ner",
    model="./bert_finetuned",
    tokenizer="./bert_finetuned",
    device=device,
    aggregation_strategy="simple"
)

# 加载CoNLL2003数据集
dataset = load_dataset("conll2003")

# 定义标签映射
label_mapping = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC"
}

def extract_entities_with_bert(tokens):
    """使用微调模型进行实体识别"""
    text = " ".join(tokens)
    predictions = ner_pipeline(text)
    
    entities = []
    for pred in predictions:
        entity_type = pred["entity_group"]
        if entity_type in label_mapping:
            entities.append([
                pred["word"],  
                entity_type    
            ])
    return text, entities

# 处理每个数据集分割
for split in ["train", "validation", "test"]:
    results = []
    for example in tqdm(dataset[split], desc=f"Processing {split} set"):
        tokens = example["tokens"]
        
        try:
            input_text, entities = extract_entities_with_bert(tokens)
            
            results.append({
                "input_text": input_text,
                "entities": entities  
            })
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            continue
    
    # 保存到JSON文件
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"{split}集处理完成，已保存到 {output_path}")

print("所有处理完成！结果已保存到", output_dir)