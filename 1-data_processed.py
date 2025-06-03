import os
import spacy
import json
from datasets import load_dataset
from tqdm import tqdm

# 确保输出目录存在
output_dir = "conll2003_processed"
os.makedirs(output_dir, exist_ok=True)

# 加载模型和数据集
nlp = spacy.load("en_core_web_lg")
dataset = load_dataset("conll2003")

# 定义实体类型映射
entity_mapping = {
    # 直接对应
    "ORG": "ORG",
    "PERSON": "PER",
    "GPE": "LOC",
    "LOC": "LOC",
    # 争议类型处理
    "NORP": "ORG",
    "FAC": "LOC",
    # 其余全部归为MISC
    "PRODUCT": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "PERCENT": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC"
}

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        ent_type = entity_mapping.get(ent.label_, "MISC")
        entities.append((ent.text, ent_type))
    return entities

# 处理每个数据集分割
for split in ["train","validation", "test"]:
    results = []
    for example in tqdm(dataset[split], desc=f"Processing {split} set"):
        text = " ".join(example["tokens"])
        entities = extract_entities(text)
        if entities:
            results.append({
                "input_text": text,
                "entities": entities
            })
    
    # 保存到单独的文件
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"{split}集处理完成，已保存到 {output_path}，共 {len(results)} 条记录。")

print("所有处理完成！三个分割数据集已分别保存到", output_dir, "目录下。")