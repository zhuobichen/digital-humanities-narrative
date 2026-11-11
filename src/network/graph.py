"""人物社交网络构建与分析"""

from typing import Any, List
from collections import Counter
import networkx as nx

from ..models import Relation, NetworkStats

class CharacterNetworkBuilder:
    """人物社交网络构建器

    基于关系数据构建人物社交网络图，并提供中心性分析、社区检测等功能。
    """

    def __init__(self):
        self.G: nx.Graph | None = None

    def build_network(
        self,
        relations: List[Relation],
        extra_characters: list[str] | None = None,
        set_as_current: bool = True,
    ) -> nx.Graph:
        """从关系列表构建社交网络

        Args:
            relations: RelationshipExtractor 输出的 Relation 对象列表
            extra_characters: 额外要放进图里的孤立人物（没有关系的角色，
                不传就会在可视化里凭空消失，人数和画像对不上）
            set_as_current: 是否把结果写回 self.G。时间轴快照这类"临时图"必须传
                False——否则最后一次快照会把主图覆盖成残缺版本（可视化跟着错）

        Returns:
            NetworkX 图对象
        """
        G = nx.Graph()

        for name in extra_characters or []:
            if name and name not in G:
                G.add_node(name, type="character", locations=[])

        for rel in relations:
            source = rel.source
            target = rel.target
            rel_type = rel.type
            sentiment = rel.sentiment
            location = rel.location # 新增：提取地点

            # 处理节点
            if source not in G:
                G.add_node(source, type='character', locations=[])
            else:
                G.nodes[source]['type'] = 'character'
                if 'locations' not in G.nodes[source]:
                    G.nodes[source]['locations'] = []
                    
            if target not in G:
                G.add_node(target, type='character', locations=[])
            else:
                G.nodes[target]['type'] = 'character'
                if 'locations' not in G.nodes[target]:
                    G.nodes[target]['locations'] = []
            
            # 处理地点节点 (如果存在且不是"未知")
            if location and location != "未知":
                if location not in G:
                    G.add_node(location, type='location')
                # 建立人物到地点的关联边
                G.add_edge(source, location, weight=1, type="location_link", sentiment="neutral", context="")
                G.add_edge(target, location, weight=1, type="location_link", sentiment="neutral", context="")

            if G.has_edge(source, target):
                # 累加权重
                G[source][target]["weight"] += 1
                # 追加关系类型和情感倾向
                if "types" not in G[source][target]:
                    G[source][target]["types"] = []
                if rel_type not in G[source][target]["types"]:
                    G[source][target]["types"].append(rel_type)
                # 简单平均情感倾向 (这里假设 neutral=0.5, positive=1.0, negative=0.0，实际存储的是字符串，我们可以保留最后一次或投票)
                # 为了简单起见，我们保留一个列表或最新状态，这里用列表记录所有互动上下文
                if "contexts" not in G[source][target]:
                    G[source][target]["contexts"] = []
                G[source][target]["contexts"].append({"snippet": rel.context, "sentiment": sentiment, "location": location})
                
                # 统计常去地点
                if location and location != "未知":
                    G.nodes[source]["locations"].append(location)
                    G.nodes[target]["locations"].append(location)
            else:
                if location and location != "未知":
                    G.nodes[source]["locations"].append(location)
                    G.nodes[target]["locations"].append(location)
                G.add_edge(
                    source,
                    target,
                    weight=1,
                    types=[rel_type],
                    sentiment=sentiment,
                    context_snippet=rel.context, # 兼容旧前端
                    contexts=[{"snippet": rel.context, "sentiment": sentiment, "location": location}]
                )

        # 整理节点的地点（取最常出现的 Top 3）
        for node in G.nodes():
            if "locations" in G.nodes[node]:
                locs = G.nodes[node]["locations"]
                if locs:
                    most_common = [loc for loc, count in Counter(locs).most_common(3)]
                    G.nodes[node]["top_locations"] = most_common
                else:
                    G.nodes[node]["top_locations"] = []
            else:
                G.nodes[node]["top_locations"] = []

        if set_as_current:
            self.G = G
        return G

    def analyze_network(self) -> NetworkStats:
        """分析网络特征

        Returns:
            NetworkStats 对象，包含网络指标
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            return NetworkStats(
                num_characters=0,
                num_relations=0,
                density=0.0
            )

        G = self.G

        # 为了不污染社区检测和中心性计算，我们需要过滤掉 'location' 类型的节点
        # 创建一个只包含人物节点的子图
        character_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'character']
        person_graph = G.subgraph(character_nodes)

        # 度中心性（谁认识的人多）
        degree_cent = nx.degree_centrality(person_graph)

        # 介数中心性（谁是信息桥梁）
        betweenness_cent = nx.betweenness_centrality(person_graph)

        # 社区检测
        try:
            communities_raw = list(nx.community.greedy_modularity_communities(person_graph))
            communities = [list(c) for c in communities_raw]
        except Exception:
            communities = []

        # 关键人物
        main_char = max(degree_cent, key=degree_cent.get) if degree_cent else None
        bridge_char = max(betweenness_cent, key=betweenness_cent.get) if betweenness_cent else None

        return NetworkStats(
            num_characters=G.number_of_nodes(),
            num_relations=G.number_of_edges(),
            density=nx.density(G),
            main_character=main_char,
            bridge_character=bridge_char,
            communities=communities,
            degree_centrality=degree_cent,
            betweenness_centrality=betweenness_cent
        )

    def get_ego_network(self, character: str, radius: int = 1) -> nx.Graph:
        """获取某人物的自我网络（邻域子图）

        Args:
            character: 人物名
            radius: 邻域半径（跳数）

        Returns:
            子图
        """
        if self.G is None:
            raise ValueError("请先调用 build_network")

        nodes = set([character])
        current_layer = set([character])

        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                next_layer.update(self.G.neighbors(node))
            nodes.update(next_layer)
            current_layer = next_layer

        return self.G.subgraph(nodes)

    def get_character_stats(self, character: str) -> dict[str, Any]:
        """获取单个人物的网络统计

        Args:
            character: 人物名

        Returns:
            人物网络指标
        """
        if self.G is None:
            raise ValueError("请先调用 build_network")

        if character not in self.G:
            return {"error": f"人物 '{character}' 不在网络中"}

        return {
            "name": character,
            "degree": self.G.degree(character),
            "neighbors": list(self.G.neighbors(character)),
            "num_neighbors": len(list(self.G.neighbors(character))),
            "edge_types": self._summarize_edge_types(character),
        }

    def _summarize_edge_types(self, character: str) -> Counter:
        """汇总某人物所有边的类型"""
        types_counter: Counter = Counter()
        if self.G is None:
            return types_counter

        for neighbor in self.G.neighbors(character):
            edge_data = self.G[character][neighbor]
            for t in edge_data.get("types", []):
                types_counter[t] += 1

        return types_counter
