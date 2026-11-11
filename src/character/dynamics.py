"""人物关系动态分析器

分析人物关系随时间的变化趋势，识别关系转折点。
"""

from typing import Any
from collections import defaultdict


class RelationshipDynamicsAnalyzer:
    """关系动态分析器

    分析人物关系的演变轨迹，包括情感趋势、关系类型变化和转折点检测。
    """

    def __init__(self):
        pass

    def analyze_evolution(
        self, relations_over_time: dict[tuple[str, str], list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """分析关系随时间的演变

        Args:
            relations_over_time: {(人物A, 人物B): [按时间排序的互动列表]}

        Returns:
            关系演变分析结果
        """
        evolution = {}

        for (char1, char2), interactions in relations_over_time.items():
            # 按时间排序
            sorted_interactions = sorted(
                interactions, key=lambda x: x.get("time", {}).get("timestamp", 0) if x.get("time") else 0
            )

            # 提取情感趋势
            sentiment_trend = [
                i.get("sentiment", 0.5) for i in sorted_interactions
            ]

            relationship_key = f"{char1}-{char2}"

            evolution[relationship_key] = {
                "characters": (char1, char2),
                "num_interactions": len(sorted_interactions),
                "interactions": sorted_interactions,
                "sentiment_trend": sentiment_trend,
                "avg_sentiment": (
                    sum(sentiment_trend) / len(sentiment_trend)
                    if sentiment_trend else 0.5
                ),
                "relationship_type": self._classify_relationship(sorted_interactions),
                "turning_points": self._identify_turning_points(sorted_interactions),
                "trend": self._detect_trend(sentiment_trend),
            }

        return evolution

    def analyze_from_relations(
        self, relations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """从关系列表直接分析

        Args:
            relations: RelationshipExtractor 输出的关系列表；元素可以是
                Relation 对象，也可以是已经 model_dump() 过的 dict。

        Returns:
            关系演变分析结果
        """
        # 按人物对分组
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for rel in relations:
            # 上游返回 Pydantic 对象，测试与旧调用方传 dict；统一成 field 取值，
            # 否则这里会直接 TypeError: 'Relation' object is not subscriptable。
            def _field(obj, name, default=""):
                if isinstance(obj, dict):
                    return obj.get(name, default)
                return getattr(obj, name, default)

            item = {
                "source": _field(rel, "source"),
                "target": _field(rel, "target"),
                "type": _field(rel, "type", "unknown"),
                "context": _field(rel, "context"),
                "context_snippet": _field(rel, "context_snippet"),
                "position": _field(rel, "position", 0),
            }
            pair = tuple(sorted([item["source"], item["target"]]))
            grouped[pair].append(item)

        return self.analyze_evolution(grouped)

    def _classify_relationship(self, interactions: list[dict[str, Any]]) -> str:
        """分类关系类型"""
        if not interactions:
            return "unknown"

        collab_count = sum(1 for i in interactions if i.get("type") == "collaborative")
        conflict_count = sum(1 for i in interactions if i.get("type") == "conflict")
        social_count = sum(1 for i in interactions if i.get("type") == "social")
        hierarchical_count = sum(1 for i in interactions if i.get("type") == "hierarchical")

        total = len(interactions)

        if hierarchical_count / total > 0.4:
            return "上下级"
        elif conflict_count / total > 0.4:
            return "对立关系"
        elif social_count / total > 0.5:
            return "朋友"
        elif collab_count / total > 0.6:
            return "同事/合作伙伴"
        elif collab_count > 0 and social_count > 0:
            return "亲密的同事"
        else:
            return "泛泛之交"

    def _identify_turning_points(
        self, interactions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """识别关系转折点"""
        turning_points = []

        for i in range(1, len(interactions)):
            prev = interactions[i - 1]
            curr = interactions[i]

            sentiment_change = abs(
                curr.get("sentiment", 0.5) - prev.get("sentiment", 0.5)
            )

            # 情感突变
            if sentiment_change > 0.5:
                turning_points.append({
                    "index": i,
                    "type": "sentiment_shift",
                    "description": (
                        f"情感从 {prev.get('sentiment'):.2f} "
                        f"变为 {curr.get('sentiment'):.2f}"
                    ),
                    "prev_event": prev,
                    "curr_event": curr,
                })

            # 关系类型变化
            if curr.get("type") != prev.get("type"):
                turning_points.append({
                    "index": i,
                    "type": "relationship_change",
                    "description": (
                        f"关系从 {prev.get('type')} 变为 {curr.get('type')}"
                    ),
                    "prev_event": prev,
                    "curr_event": curr,
                })

        return turning_points

    def _detect_trend(self, sentiment_values: list[float]) -> str:
        """检测情感趋势

        Returns:
            "improving", "declining", "stable", "fluctuating"
        """
        if len(sentiment_values) < 2:
            return "stable"

        # 简单线性回归方向
        n = len(sentiment_values)
        x_mean = (n - 1) / 2
        y_mean = sum(sentiment_values) / n

        num = sum((i - x_mean) * (sentiment_values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        if den == 0:
            return "stable"

        slope = num / den

        # 检查波动性
        variance = sum((s - y_mean) ** 2 for s in sentiment_values) / n

        if variance > 0.1:
            if slope > 0.02:
                return "improving"
            elif slope < -0.02:
                return "declining"
            else:
                return "fluctuating"
        else:
            return "stable"

    def get_relationship_summary(
        self, evolution: dict[str, Any]
    ) -> list[dict[str, str]]:
        """生成关系演变的可读摘要

        Args:
            evolution: analyze_evolution 的输出

        Returns:
            摘要列表
        """
        summaries = []
        for rel_key, data in evolution.items():
            char1, char2 = data["characters"]
            summaries.append({
                "pair": f"{char1} ↔ {char2}",
                "type": data["relationship_type"],
                "interactions": str(data["num_interactions"]),
                "trend": {
                    "improving": "逐渐改善",
                    "declining": "逐渐恶化",
                    "stable": "保持稳定",
                    "fluctuating": "波动变化",
                }.get(data["trend"], data["trend"]),
                "turning_points": str(len(data["turning_points"])),
            })

        return summaries
