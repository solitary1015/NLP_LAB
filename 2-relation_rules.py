import os
import json
import spacy
from spacy.matcher import Matcher, DependencyMatcher
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
    
        self.matcher = Matcher(self.nlp.vocab)
        self.dep_matcher = DependencyMatcher(self.nlp.vocab)
        
        self._setup_patterns()
        
        self.triplets = []
        
    def _setup_patterns(self):
        
        # 1. 基于依存关系的模式
        # Person works for Organization
        work_pattern = [
            {
                "RIGHT_ID": "person",
                "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"}
            },
            {
                "LEFT_ID": "person",
                "REL_OP": ">",
                "RIGHT_ID": "work_verb",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["work", "serve", "join", "head", "lead", "manage"]}}
            },
            {
                "LEFT_ID": "work_verb",
                "REL_OP": ">",
                "RIGHT_ID": "org",
                "RIGHT_ATTRS": {"ENT_TYPE": "ORG"}
            }
        ]
        self.dep_matcher.add("WORK_FOR", [work_pattern])
        
        # Person travels to Location
        travel_pattern = [
            {
                "RIGHT_ID": "person",
                "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"}
            },
            {
                "LEFT_ID": "person",
                "REL_OP": ">",
                "RIGHT_ID": "travel_verb",
                "RIGHT_ATTRS": {"LEMMA": {"IN": ["go", "travel", "head", "visit", "arrive", "leave", "fly"]}}
            },
            {
                "LEFT_ID": "travel_verb",
                "REL_OP": ">",
                "RIGHT_ID": "location",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}
            }
        ]
        self.dep_matcher.add("TRAVEL_TO", [travel_pattern])
        
        # Organization located in Location
        location_pattern = [
            {
                "RIGHT_ID": "org",
                "RIGHT_ATTRS": {"ENT_TYPE": "ORG"}
            },
            {
                "LEFT_ID": "org",
                "REL_OP": ">",
                "RIGHT_ID": "prep",
                "RIGHT_ATTRS": {"DEP": "prep", "TEXT": "in"}
            },
            {
                "LEFT_ID": "prep",
                "REL_OP": ">",
                "RIGHT_ID": "location",
                "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}
            }
        ]
        self.dep_matcher.add("LOCATED_IN", [location_pattern])
        
        # 2. 基于词汇模式的匹配
        # Person's Organization
        possession_pattern = [
            {"ENT_TYPE": "PERSON"},
            {"TEXT": "'s"},
            {"ENT_TYPE": "ORG"}
        ]
        self.matcher.add("AFFILIATED_WITH", [possession_pattern])
        
        # Organization official Person
        official_pattern = [
            {"ENT_TYPE": "ORG"},
            {"LEMMA": {"IN": ["official", "representative", "spokesman", "leader", "head", "director", "president", "CEO"]}},
            {"ENT_TYPE": "PERSON"}
        ]
        self.matcher.add("OFFICIAL_OF", [official_pattern])
        
        # Person from Location
        from_pattern = [
            {"ENT_TYPE": "PERSON"},
            {"TEXT": "from"},
            {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}
        ]
        self.matcher.add("FROM", [from_pattern])
    
    #从单个文本中抽取关系
    def extract_relations_from_text(self, text, entities):
        doc = self.nlp(text)
        relations = []
        
        # 文本到类型的实体映射
        entity_map = {}
        for ent_text, ent_type in entities:
            entity_map[ent_text.lower()] = ent_type
        
        # 使用依存关系匹配器
        dep_matches = self.dep_matcher(doc)
        for match_id, token_ids in dep_matches:
            relation_label = self.nlp.vocab.strings[match_id]
            tokens = [doc[token_id] for token_id in token_ids]
                
            if relation_label == "WORK_FOR":
                person_token = tokens[0]
                org_token = tokens[2]
                if person_token.ent_type_ == "PERSON" and org_token.ent_type_ == "ORG":
                    relations.append((person_token.text, "works_for", org_token.text))
                    relations.append((person_token.text, "affiliated_with", org_token.text))
                
            elif relation_label == "TRAVEL_TO":
                person_token = tokens[0]
                loc_token = tokens[2]
                if person_token.ent_type_ == "PERSON" and loc_token.ent_type_ in ["GPE", "LOC"]:
                    relations.append((person_token.text, "travels_to", loc_token.text))
                    relations.append((person_token.text, "heads_for", loc_token.text))
                
            elif relation_label == "LOCATED_IN":
                org_token = tokens[0]
                loc_token = tokens[2]
                if org_token.ent_type_ == "ORG" and loc_token.ent_type_ in ["GPE", "LOC"]:
                    relations.append((org_token.text, "located_in", loc_token.text))
        
        # 使用词汇模式匹配器
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            relation_label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            if relation_label == "AFFILIATED_WITH":
                if len(span) >= 3:
                    person = span[0].text
                    org = span[2].text
                    relations.append((person, "affiliated_with", org))
            
            elif relation_label == "OFFICIAL_OF":
                if len(span) >= 3:
                    org = span[0].text
                    person = span[2].text
                    relations.append((person, "official_of", org))
                    relations.append((person, "affiliated_with", org))
            
            elif relation_label == "FROM":
                if len(span) >= 3:
                    person = span[0].text
                    location = span[2].text
                    relations.append((person, "from", location))
        
        # 基于距离的简单关系推断
        doc_entities = [(ent.text, ent.label_) for ent in doc.ents]
        for i, (ent1_text, ent1_type) in enumerate(doc_entities):
            for j, (ent2_text, ent2_type) in enumerate(doc_entities):
                if i != j:
                    # 人员与组织的关系
                    if ent1_type == "PERSON" and ent2_type == "ORG":
                        # 检查是否在句子中相邻或接近
                        ent1_pos = doc.text.find(ent1_text)
                        ent2_pos = doc.text.find(ent2_text)
                        if abs(ent1_pos - ent2_pos) < 50:  # 距离阈值
                            relations.append((ent1_text, "affiliated_with", ent2_text))
                    
                    # 人员与地点的关系
                    elif ent1_type == "PERSON" and ent2_type in ["GPE", "LOC"]:
                        # 查找移动动词
                        for token in doc:
                            if token.lemma_ in ["go", "travel", "head", "visit", "arrive"]:
                                relations.append((ent1_text, "travels_to", ent2_text))
                                break
        
        return relations
    
    def process_dataset(self, data_file):
        print(f"Processing {data_file}...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_relations = []
        
        for item in tqdm(data, desc="Extracting relations"):
            text = item['input_text']
            entities = item['entities']
            
            # 抽取关系
            relations = self.extract_relations_from_text(text, entities)
            
            # 添加到总列表
            all_relations.extend(relations)
        
        return all_relations
    
    def save_knowledge_graph(self, relations, output_file):
        # 保存为CSV
        csv_file = output_file.replace('.json', '.csv')
        df = pd.DataFrame(relations,columns=["subject", "relation", "object"])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Knowledge graph saved to {csv_file}")
        
        # 保存为JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(relations, f, indent=2, ensure_ascii=False)
        print(f"Knowledge graph saved to {output_file}")
        
        return df
    
    def visualize_knowledge_graph(self, relations_df, output_file="knowledge_graph.png", max_nodes=50):
        print("Creating knowledge graph visualization")
        
        G = nx.DiGraph()
        
        # 统计关系频次，只显示最重要的关系
        relation_counts = defaultdict(int)
        for _, row in relations_df.iterrows():
            relation_counts[(row['subject'], row['relation'], row['object'])] += 1
        
        # 按频次排序，取前max_nodes个
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        
        # 添加节点和边
        for (subj, rel, obj), count in sorted_relations:
            G.add_edge(subj, obj, relation=rel, weight=count)
        
        # 设置图形大小
        plt.figure(figsize=(15, 10))
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        node_colors = []
        for node in G.nodes():
            # 根据节点类型设置颜色（简单启发式）
            if any(word in node.lower() for word in ['corp', 'inc', 'ltd', 'company', 'org']):
                node_colors.append('lightblue')  # 组织
            elif node.isupper() and len(node) <= 5:
                node_colors.append('lightgreen')  # 可能是缩写/地名
            else:
                node_colors.append('lightcoral')  # 人名
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # 添加边标签（关系）
        edge_labels = {}
        for subj, obj, data in G.edges(data=True):
            edge_labels[(subj, obj)] = data['relation']
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Knowledge graph visualization saved to {output_file}")
        
        # 打印图统计信息
        print(f"\nGraph Statistics:")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Number of unique relations: {len(set(data['relation'] for _, _, data in G.edges(data=True)))}")


def main():
    # 创建关系抽取器
    extractor = RelationExtractor()
    
    # 处理数据集
    data_dir = "conll2003_bert_processed"
    output_dir = "knowledge_graph_output"
    os.makedirs(output_dir, exist_ok=True)
    
    all_relations = []
    
    # 处理每个数据集分割
    for split in ["train", "validation", "test"]:
        data_file = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(data_file):
            relations = extractor.process_dataset(data_file)
            all_relations.extend(relations)
            print(f"Extracted {len(relations)} relations from {split} set")
    
    print(f"\nTotal relations extracted: {len(all_relations)}")
    
    # 去重
    unique_relations = []
    seen = set()
    for rel in all_relations:
        key = (rel[0], rel[1], rel[2])
        if key not in seen:
            seen.add(key)
            unique_relations.append(rel)
    
    print(f"Unique relations after deduplication: {len(unique_relations)}")
    
    # 保存知识图谱
    output_file = os.path.join(output_dir, "knowledge_graph.json")
    df = extractor.save_knowledge_graph(unique_relations, output_file)
    
    # 可视化知识图谱
    viz_file = os.path.join(output_dir, "knowledge_graph_visualization.png")
    extractor.visualize_knowledge_graph(df, viz_file)
    
    # 打印一些统计信息
    print(f"\nRelation Types:")
    relation_counts = df['relation'].value_counts()
    print(relation_counts.head(10))
    
    print(f"\nMost Common Entities:")
    subject_counts = df['subject'].value_counts()
    object_counts = df['object'].value_counts()
    print("Subjects:", subject_counts.head(5).to_dict())
    print("Objects:", object_counts.head(5).to_dict())


if __name__ == "__main__":
    main()