import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from seqeval.metrics import classification_report
from tqdm import tqdm
import torch

# 加载微调模型
device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline(
    "ner",
    model="./bert_finetuned",
    tokenizer="./bert_finetuned",
    device=device,
    aggregation_strategy="simple"
)

# 加载测试集
dataset = load_dataset("conll2003")["test"]

# 标签ID到类型的映射（与训练时一致）
id2label = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC"
}

def evaluate_bert_model(pipeline, dataset, id2label):
    """
    评估微调BERT模型并返回评估指标
    
    参数:
        pipeline: 加载的BERT pipeline
        dataset: CoNLL2003测试集
        id2label: 标签ID到类型的映射
    """
    true_labels = []
    pred_labels = []
    
    for example in tqdm(dataset, desc="Evaluating BERT Model"):
        tokens = example["tokens"]
        gold_ner_tags = example["ner_tags"]
        
        # 处理真实标签
        gold_labels = [id2label[tag] for tag in gold_ner_tags]
        true_labels.append(gold_labels)
        
        # 初始化预测标签为全'O'
        pred_labels_seq = ["O"] * len(tokens)
        
        # 使用BERT模型预测
        text = " ".join(tokens)
        predictions = pipeline(text)
        
        # 处理BERT预测的实体
        for pred in predictions:
            entity_type = pred["entity_group"]
            if entity_type == "O":
                continue
                
            # 找到实体在原始token序列中的位置
            ent_tokens = pred["word"].split()
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
                        pred_labels_seq[i] = f"B-{entity_type}"
                    else:
                        pred_labels_seq[i] = f"I-{entity_type}"
        
        pred_labels.append(pred_labels_seq)
    
    # 生成评估报告
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    return report_df

# 运行评估
metrics_df = evaluate_bert_model(ner_pipeline, dataset, id2label)

# 保存评估结果
metrics_df.to_csv("bert_evaluation_metrics.csv", index=True)

print("BERT模型评估结果已保存到 bert_evaluation_metrics.csv")
print("\n评估指标概览:")
print(metrics_df)