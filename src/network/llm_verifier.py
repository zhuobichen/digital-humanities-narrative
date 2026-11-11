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

# 复核强度：
#   strict —— 只保留模型判成立的边（模型当前默认，association 也照删）
#   normal —— association 一律保留，其余以模型判定为准
#   loose  —— 只剔除"明显误判"（数值/假设语气/转述），association 与其余保留
STRENGTHS = ("strict", "normal", "loose")

# loose 档愿意因哪些理由剔边（理由文本来自模型，按关键词近似）
LOOSE_DROP_PATTERNS = ("数值", "不是人物", "假设", "如果", "转述", "比喻", "零是")


def should_keep(cand: dict, verdict: dict | None, strength: str = "strict") -> bool:
    """按复核强度决定是否保留一条边。

    Args:
        cand: 候选边
        verdict: 模型的判定（None 表示没有判定，按保留处理）
        strength: strict / normal / loose

    Returns:
        True 保留，False 剔除
    """
    if strength not in STRENGTHS:
        raise ValueError(f"未知的复核强度: {strength!r}，可选 {STRENGTHS}")
    if verdict is None:
        return True
    valid = verdict.get("valid")
    if valid is not False:
        return True

    reason = str(verdict.get("reason", ""))
    if strength == "strict":
        return False
    if strength == "normal":
        # 只放行 association：泛关系即使模型说不成立也留下
        return cand.get("type") == "association"
    # loose：只有明显误判才删
    return not any(p in reason for p in LOOSE_DROP_PATTERNS)


def review_candidates(
    extractor, candidates: list[dict], max_items: int = 200, strength: str = "strict"
) -> list[dict]:
    """用 LLM 复核候选边。

    Args:
        extractor: LLMRelationshipExtractor 实例（没有 key 时 available=False）
        candidates: [{source, target, type, weight, evidence}]
        max_items: 最多复核多少条（按输入顺序，调用方应已按权重排序）
        strength: 复核强度 strict/normal/loose，见 `should_keep`

    Returns:
        新的候选列表：被判定不成立的剔除，类型被纠正的改类型，
        每条附加 reviewed / reason 字段。extractor 不可用时原样返回。
    """
    if strength not in STRENGTHS:
        raise ValueError(f"未知的复核强度: {strength!r}，可选 {STRENGTHS}")

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
        if not should_keep(cand, verdict, strength):
            continue
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
