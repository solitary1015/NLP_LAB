import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import Model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Step 1: 读取数据，准备三元组
df = pd.read_csv("./knowledge_graph_output/knowledge_graph.csv")
triples = list(df.itertuples(index=False, name=None))
triples = np.array(triples, dtype=str)

# Step 2: 创建 TriplesFactory
tf = TriplesFactory.from_labeled_triples(triples)

# Step 3: 划分数据集
training_tf, testing_tf, validation_tf = tf.split(
    ratios=[0.7, 0.2, 0.1],
    random_state=42,
    method='cleanup',
)

###### 按顺序运行TransE、ComplEx、DistMult模型
###### 不然CSV的覆写会导致数据丢失
model_name= 'TransE'  # 可选 'TransE' 或 'DistMult'
###### 可选 'TransE','ComplEx' 或 'DistMult'

# Step 4: 训练模型
print(f"\n=== 开始训练 {model_name} 模型 ===")
result = pipeline(
    training=training_tf,
    testing=testing_tf,
    validation=validation_tf,
    model=model_name,            
    training_kwargs=dict(num_epochs=1000, use_tqdm_batch=False),
)
print(f"=== {model_name} 模型训练完成 ===")

# Step 5: 保存模型和训练信息
save_path = Path(f'./kl_saved_models/{model_name}')
save_path.mkdir(parents=True, exist_ok=True)
result.save_to_directory(save_path)
# 手动构建一个兼容的 configuration.json 内容
config_dict = {
    "model": model_name,
    "dataset": "n/a",  # 因为我们是手动构造的 TriplesFactory
    "training": str((save_path / 'training_triples').absolute()),
    "testing": str((save_path / 'testing_triples').absolute()),
    "validation": str((save_path / 'validation_triples').absolute()),
    "random_seed": 42,
    "pykeen_version": "1.11.0"
}
with open(save_path / 'configuration.json', 'w', encoding='utf-8') as f:
    json.dump(config_dict, f, indent=4)
# 手动保存验证/测试三元组
validation_tf.to_path_binary(save_path / 'validation_triples')
testing_tf.to_path_binary(save_path / 'testing_triples')
print(f"模型已保存到 {save_path}")

# Step 6: 模型评估
model: Model = result.model  
# === 链接预测评估指标 ===
print("\n=== 链接预测评估指标 ===")

print(f"MRR: {result.get_metric('mean_reciprocal_rank')}")
print(f"Hits@1: {result.get_metric('hits_at_1')}")
print(f"Hits@3: {result.get_metric('hits_at_3')}")
print(f"Hits@10: {result.get_metric('hits_at_10')}")
print(f"MeanRank: {result.get_metric('mean_rank')}")

# === 三元组分类评估 ===
print("\n=== 三元组分类评估 ===")

# 基础准备
positive_triples = testing_tf.mapped_triples
num_pos = len(positive_triples)
num_entities = model.num_entities

# 构造负样本（替换头和尾），并过滤掉正样本
positive_set = set(map(tuple, positive_triples.cpu().numpy()))

def sample_negatives(positive_triples, num_entities, positive_set):
    negs = []
    for i in range(len(positive_triples)):
        h, r, t = positive_triples[i].tolist()
        # 替换头实体
        while True:
            new_h = np.random.randint(0, num_entities)
            if (new_h, r, t) not in positive_set:
                negs.append([new_h, r, t])
                break
        # 替换尾实体
        while True:
            new_t = np.random.randint(0, num_entities)
            if (h, r, new_t) not in positive_set:
                negs.append([h, r, new_t])
                break
    return torch.tensor(negs, dtype=positive_triples.dtype)

neg_triples = sample_negatives(positive_triples, num_entities, positive_set)
num_neg = len(neg_triples)

# 拼接正负样本
all_triples = torch.cat([positive_triples, neg_triples], dim=0)
labels = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])

# 模型打分
with torch.no_grad():
    scores = model.score_hrt(all_triples).cpu().numpy()

# 使用ROC曲线自动选择最佳阈值
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

preds = (scores >= best_threshold).astype(int)

# 评估指标
acc = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
auc = roc_auc_score(labels, scores)

# 输出评估结果
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
# print("\n=== 三元组分类评估 ===")

# # 基础准备
# positive_triples = testing_tf.mapped_triples
# num_pos = len(positive_triples)
# num_entities = model.num_entities

# # 构造负样本（替换头和尾）
# # 替换头实体
# neg_heads = positive_triples.clone()
# rand_heads = torch.randint(low=0, high=num_entities, size=(num_pos,))
# neg_heads[:, 0] = rand_heads

# # 替换尾实体
# neg_tails = positive_triples.clone()
# rand_tails = torch.randint(low=0, high=num_entities, size=(num_pos,))
# neg_tails[:, 2] = rand_tails

# # 合并所有负样本
# neg_triples = torch.cat([neg_heads, neg_tails], dim=0)
# num_neg = len(neg_triples)

# # 拼接正负样本
# all_triples = torch.cat([positive_triples, neg_triples], dim=0)
# labels = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])

# # 模型打分
# with torch.no_grad():
#     scores = model.score_hrt(all_triples).cpu().numpy()

# # 简单分类评估
# threshold = np.median(scores)
# preds = (scores >= threshold).astype(int)

# # 评估指标
# acc = accuracy_score(labels, preds)
# precision = precision_score(labels, preds)
# recall = recall_score(labels, preds)
# f1 = f1_score(labels, preds)
# auc = roc_auc_score(labels, scores)

# # 输出评估结果
# print(f"Accuracy : {acc:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1 Score : {f1:.4f}")
# print(f"ROC AUC  : {auc:.4f}")


# Step 6: 保存评估结果
# === 保存链接预测评估结果 ===
link_metrics = {
    'Model': model_name,
    'MRR': result.get_metric('mean_reciprocal_rank'),
    'Hits@1': result.get_metric('hits_at_1'),
    'Hits@3': result.get_metric('hits_at_3'),
    'Hits@10': result.get_metric('hits_at_10'),
    'MeanRank': result.get_metric('mean_rank')
}
link_df = pd.DataFrame([link_metrics])
if model_name == 'TransE':
    link_df.to_csv('3-link_prediction_results.csv', index=False)
else:
    link_df.to_csv('3-link_prediction_results.csv', mode='a',header=False,index=False)
print("链接预测评估结果已保存到 3-link_prediction_results.csv")

# === 保存三元组分类评估结果 ===
triple_cls_metrics = {
    'Model': model_name,
    'Accuracy': acc,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': auc
}
triple_df = pd.DataFrame([triple_cls_metrics])
if model_name == 'TransE':
    triple_df.to_csv('3-triple_classification_results.csv', index=False)
else:
    triple_df.to_csv('3-triple_classification_results.csv', mode='a',header=False,index=False)
print("三元组分类评估结果已保存到 3-triple_classification_results.csv")       



