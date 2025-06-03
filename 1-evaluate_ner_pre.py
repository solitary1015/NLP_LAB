import pandas as pd
import spacy
from datasets import load_dataset
from seqeval.metrics import classification_report
from tqdm import tqdm

# 加载模型和数据
nlp = spacy.load("en_core_web_lg")
dataset = load_dataset("conll2003")["test"]


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

# 标签ID到类型的映射
id2label = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC"
}

def evaluate_spacy_model(nlp, dataset, entity_mapping, id2label):
    """
    仅评估Spacy NER模型并返回评估指标
    
    参数:
        nlp: 加载的Spacy模型
        dataset: CoNLL2003测试集
        entity_mapping: 实体类型映射
        id2label: 标签ID到类型的映射
    """
    true_labels = []
    pred_labels = []
    
    for example in tqdm(dataset, desc="Evaluating"):
        tokens = example["tokens"]
        gold_ner_tags = example["ner_tags"]
        
        # 处理真实标签
        gold_labels = [id2label[tag] for tag in gold_ner_tags]
        true_labels.append(gold_labels)
        
        # 使用Spacy模型预测
        doc = nlp(" ".join(tokens))
        
        # 初始化预测标签为全'O'
        pred_labels_seq = ["O"] * len(tokens)
        
        # 处理Spacy预测的实体
        for ent in doc.ents:
            mapped_label = entity_mapping.get(ent.label_, "O")
            if mapped_label == "O":
                continue
                
            # 找到实体在原始token序列中的位置
            ent_tokens = ent.text.split()
            start_pos = None
            end_pos = None
            
            # 对齐算法
            for i in range(len(tokens)):
                if tokens[i] == ent_tokens[0]:
                    match = True
                    for j in range(len(ent_tokens)):
                        if i + j >= len(tokens) or tokens[i + j] != ent_tokens[j]:
                            match = False
                            break
                    if match:
                        start_pos = i
                        end_pos = i + len(ent_tokens)
                        break
            
            if start_pos is not None:
                # 标记实体
                for i in range(start_pos, end_pos):
                    if i == start_pos:
                        pred_labels_seq[i] = f"B-{mapped_label}"
                    else:
                        pred_labels_seq[i] = f"I-{mapped_label}"
        
        pred_labels.append(pred_labels_seq)
    
    # 生成评估报告并转换为DataFrame
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    return report_df

# 运行评估并保存指标
metrics_df = evaluate_spacy_model(nlp, dataset, entity_mapping, id2label)
metrics_df.to_csv("evaluation_metrics.csv", index=True)

print("Evaluation metrics saved to evaluation_metrics.csv")
print("\nEvaluation Metrics Summary:")
print(metrics_df)