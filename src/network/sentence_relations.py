"""句内共现 + 谓词关系类型（规则版 P1-b）。

背景：原来 `RelationshipExtractor.extract_relations` 是"100 字滑动窗口内两两连边"，
步长 20、窗口 100，等于把每个人的邻居都连上，结果 85% 的边是 `association`、
密度 0.56——那是共现网络，不是关系网络。

这里改成：
- 以**句子**为单位（句末标点切分）
- 同一句里出现的人物两两连边，边权 = 共现句数（弱边可以按阈值丢弃）
- 关系类型从**句子里的谓词**推断（说/问/骂/爱/恨…），推断不出才退回 association
- 每条边保留触发它的原句作为证据
"""
from __future__ import annotations

import re

SENTENCE_END = re.compile(r"[。！？!?…\n]+")

# 谓词 → 关系类型。按顺序匹配，越具体的放越前面。
PREDICATE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("hostile", ("骂", "打", "杀", "恨", "怒", "斥", "吵", "撕", "偷袭", "威胁", "报复", "攻击", "砍", "刺")),
    ("affection", ("爱", "喜欢", "思念", "想", "吻", "抱", "牵", "脸红", "心动")),
    ("hierarchical", ("命令", "汇报", "请示", "吩咐", "管", "训", "要求", "指派", "安排", "任命")),
    ("collaborative", ("合作", "帮", "协助", "配合", "一起", "共同", "并肩", "搭档")),
    ("kinship", ("父亲", "母亲", "哥哥", "弟弟", "姐姐", "妹妹", "儿子", "女儿", "叔叔", "舅舅", "外公", "爷爷")),
    ("dialogic", ("说", "道", "问", "答", "喊", "叫", "笑", "叹", "告诉", "解释", "劝", "喃喃")),
    ("social", ("见", "遇见", "认识", "介绍", "邀请", "拜访", "同学", "朋友", "同伴")),
]

DEFAULT_TYPE = "association"


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，保留有内容的句子。"""
    parts = [p.strip() for p in SENTENCE_END.split(text)]
    return [p for p in parts if len(p) >= 2]


def detect_relation_type(sentence: str) -> str:
    """从句子的谓词推断关系类型；同一句命中多条规则时取最先命中的（越具体越靠前）。"""
    for rel_type, keywords in PREDICATE_RULES:
        if any(k in sentence for k in keywords):
            return rel_type
    return DEFAULT_TYPE


def extract_sentence_relations(
    text: str,
    characters: list[str],
    min_cooccur: int = 1,
) -> list[dict]:
    """句内共现建边。

    Args:
        text: 全文
        characters: 人物名单（应为已合并别名的规范名）
        min_cooccur: 至少共现多少句才保留这条边，默认 1（全留），调大可去掉偶发边

    Returns:
        [{source, target, type, weight, evidence}]，按 weight 降序
    """
    names = [c for c in characters if c]
    pairs: dict[tuple[str, str], dict] = {}

    for sentence in split_sentences(text):
        present = [n for n in names if n in sentence]
        if len(present) < 2:
            continue
        rel_type = detect_relation_type(sentence)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = tuple(sorted((a, b)))
                item = pairs.get(key)
                if item is None:
                    pairs[key] = {
                        "source": a,
                        "target": b,
                        "type": rel_type,
                        "weight": 1,
                        "evidence": sentence[:120],
                    }
                else:
                    item["weight"] += 1
                    # 类型只保留"更具体"的：association 会被后续具体谓词覆盖
                    if item["type"] == DEFAULT_TYPE and rel_type != DEFAULT_TYPE:
                        item["type"] = rel_type
                        item["evidence"] = sentence[:120]

    relations = [r for r in pairs.values() if r["weight"] >= min_cooccur]
    relations.sort(key=lambda r: (-r["weight"], r["source"], r["target"]))
    return relations
