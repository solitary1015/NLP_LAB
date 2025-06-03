import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.regularizers import LpRegularizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# 1. 设置目录与路径
save_dir = Path('./kl_saved_models/TransE')
model_path = save_dir / 'trained_model.pkl'

# 2. 加载旧三元组工厂和模型
train_tf = TriplesFactory.from_path_binary(save_dir / 'training_triples')
old_triples = train_tf.mapped_triples
old_model: TransE = torch.load(model_path)

# 3. 加载新三元组
new_df = pd.read_csv('./4.2-new_triples.csv')
new_triples = np.array(list(new_df.itertuples(index=False, name=None)), dtype=str)
new_tf = TriplesFactory.from_labeled_triples(new_triples)
new_triples_mapped = new_tf.mapped_triples

# 4. 合并所有三元组
combined_triples = np.concatenate([train_tf.triples, new_triples], axis=0)
combined_tf = TriplesFactory.from_labeled_triples(combined_triples)

# 5. 初始化模型并迁移旧权重
regularizer = LpRegularizer(p=2, weight=1e-5)
model = TransE(triples_factory=combined_tf, regularizer=regularizer)
with torch.no_grad():
    for entity, old_idx in train_tf.entity_to_id.items():
        if entity in combined_tf.entity_to_id:
            new_idx = combined_tf.entity_to_id[entity]
            model.entity_representations[0]._embeddings.weight[new_idx] = \
                old_model.entity_representations[0]._embeddings.weight[old_idx]
    for relation, old_idx in train_tf.relation_to_id.items():
        if relation in combined_tf.relation_to_id:
            new_idx = combined_tf.relation_to_id[relation]
            model.relation_representations[0]._embeddings.weight[new_idx] = \
                old_model.relation_representations[0]._embeddings.weight[old_idx]

# 6. 准备 Replay DataLoader
old_ratio = 0.8
new_ratio = 0.2
batch_size = 128

old_dataset = TensorDataset(old_triples.clone())
new_dataset = TensorDataset(torch.tensor(new_triples_mapped, dtype=torch.long))
old_loader = DataLoader(old_dataset, batch_size=int(batch_size * old_ratio), shuffle=True, drop_last=False)
new_loader = DataLoader(new_dataset, batch_size=int(batch_size * new_ratio), shuffle=True, drop_last=False)

def replay_data_loader():
    old_iter = iter(old_loader)
    new_iter = iter(new_loader)
    while True:
        try:
            old_batch = next(old_iter)[0]
            new_batch = next(new_iter)[0]
            combined_batch = torch.cat([old_batch, new_batch], dim=0)
            yield combined_batch
        except StopIteration:
            break

# 7. 负采样函数（手动生成负样本）
def negative_sampling(batch, num_entities):
    """对batch三元组，随机替换头或尾生成负样本"""
    batch_size = batch.size(0)
    negative_batch = batch.clone()
    mask = torch.rand(batch_size) < 0.5  # 一半替换头，一半替换尾

    # 替换头实体
    random_entities_head = torch.randint(0, num_entities, (mask.sum().item(),))
    negative_batch[mask, 0] = random_entities_head

    # 替换尾实体
    random_entities_tail = torch.randint(0, num_entities, ((~mask).sum().item(),))
    negative_batch[~mask, 2] = random_entities_tail

    return negative_batch

# 8. 训练设置
optimizer = Adam(params=model.get_grad_params(), lr=1e-4)

model.train()
num_epochs = 1000

# 9. 训练循环
for epoch in range(num_epochs):
    total_loss = 0.0
    data_iter = replay_data_loader()

    for batch_triples in tqdm(data_iter, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()

        # 负采样
        negative_batch = negative_sampling(batch_triples, combined_tf.num_entities)

        # 计算分数和损失
        positive_scores = model.score_hrt(batch_triples)
        negative_scores = model.score_hrt(negative_batch)
        loss = model.loss(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model, save_dir / 'trained_model_incre_replay.pkl')
print("✅增量训练完成，模型已保存。")

# 模型评估
testing_tf = TriplesFactory.from_path_binary(save_dir / 'testing_triples') 


print("\n=== 三元组分类评估 ===")

# 正样本三元组
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
model.eval()
with torch.no_grad():
    scores = model.score_hrt(all_triples).cpu().numpy()

# 使用ROC曲线自动选择最佳阈值
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

preds = (scores >= best_threshold).astype(int)

# 计算指标
acc = accuracy_score(labels, preds)
precision = precision_score(labels, preds)
recall = recall_score(labels, preds)
f1 = f1_score(labels, preds)
auc = roc_auc_score(labels, scores)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
