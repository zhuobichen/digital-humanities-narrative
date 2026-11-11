"""用 LLM 复核规则抽取出来的候选边（可选步骤，没配 key 就自动跳过）。

思路：规则版先给出"候选边 + 方向 + 类型 + 证据句"，模型只需要回答
「这条关系在证据句里成立吗？类型对不对？」，比让它从零抽全文便宜、也稳得多。

- 没有 API key：`available=False`，调用方应原样返回规则结果（不报错）
- 有 key：逐条复核，失败/超时的边保留规则结果并标记 `reviewed=False`
"""
from __future__ import annotations

import json
from typing import Any

REVIEW_PROMPT = """你在复核一条从小说文本中抽取的人物关系。

证据句：
{evidence}

抽取结果：{source} --[{rel_type}]--> {target}

请判断：
1. 这条关系在证据句里是否成立（主体与客体是否真的构成了这层关系）
2. 关系类型是否恰当（可选值：hostile, affection, hierarchical, collaborative,
   kinship, dialogic, social, association）

严格输出 JSON：{{"valid": true/false, "type": "上面之一", "reason": "不超过30字"}}
"""

VALID_TYPES = {
    "hostile", "affection", "hierarchical", "collaborative",
    "kinship", "dialogic", "social", "association",
}


def review_candidates(extractor, candidates: list[dict], max_items: int = 200) -> list[dict]:
    """用 LLM 复核候选边。

    Args:
        extractor: LLMRelationshipExtractor 实例（没有 key 时 available=False）
        candidates: [{source, target, type, weight, evidence}]
        max_items: 最多复核多少条（按输入顺序，调用方应已按权重排序）

    Returns:
        新的候选列表：被判定不成立的剔除，类型被纠正的改类型，
        每条附加 reviewed / reason 字段。extractor 不可用时原样返回。
    """
    if not getattr(extractor, "available", False):
        return [{**c, "reviewed": False, "reason": "no_llm"} for c in candidates]

    kept: list[dict] = []
    for idx, cand in enumerate(candidates):
        if idx >= max_items:
            kept.append({**cand, "reviewed": False, "reason": "over_limit"})
            continue
        prompt = REVIEW_PROMPT.format(
            evidence=cand.get("evidence", ""),
            source=cand.get("source", ""),
            target=cand.get("target", ""),
            rel_type=cand.get("type", ""),
        )
        try:
            raw = extractor._call_llm_with_retry(prompt)
            verdict = json.loads(raw)
        except Exception as exc:  # 单条失败不拖垮整批
            kept.append({**cand, "reviewed": False, "reason": f"error: {type(exc).__name__}"})
            continue

        if not isinstance(verdict, dict):
            kept.append({**cand, "reviewed": False, "reason": "bad_format"})
            continue

        reviewed = {**cand, "reviewed": True, "reason": str(verdict.get("reason", ""))[:60]}
        if verdict.get("valid") is False:
            continue  # 模型判定不成立，丢弃这条边
        new_type = str(verdict.get("type", "")).strip()
        if new_type in VALID_TYPES:
            reviewed["type"] = new_type
        kept.append(reviewed)
    return kept


def summarize_review(before: list[dict], after: list[dict]) -> dict[str, Any]:
    """复核前后的对比，供 README/日志记录。"""
    before_types: dict[str, int] = {}
    for c in before:
        before_types[c["type"]] = before_types.get(c["type"], 0) + 1
    after_types: dict[str, int] = {}
    for c in after:
        after_types[c["type"]] = after_types.get(c["type"], 0) + 1
    return {
        "candidates": len(before),
        "kept": len(after),
        "dropped": len(before) - len(after),
        "reviewed": sum(1 for c in after if c.get("reviewed")),
        "types_before": before_types,
        "types_after": after_types,
    }
