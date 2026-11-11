"""有向关系抽取：主体 —动作→ 客体（零模型，可离线）。

为什么不用 LTP/HanLP：本机实测 `huggingface.co` 不可达（curl 返回 000），
走 hf-mirror 能下到 172MB 的 LTP/small，但 LTP 4.2 与 transformers 5.x 不兼容
（`BertTokenizer has no attribute batch_encode_plus`），装回旧版又会牵连用户的 torch。
结论：在这台机器上，句法模型这条路当前走不通。

折中方案（本文件）：用
1. `jieba.posseg` 做分词 + 词性（零模型）
2. 句子里"人物 + 动词 + 人物"的**先后顺序**定方向
3. 动词 → 关系类型的词典

这不是依存句法，但对"N 对 N 谁跟谁"这类问题够用：方向来自语序，
类型来自谓词，证据仍是原句。真正的精确化要等有可用句法模型时再替换。
"""
from __future__ import annotations

import re

import jieba
import jieba.posseg as pseg

from .sentence_relations import DEFAULT_TYPE, PREDICATE_RULES, split_sentences

# 动词 → 关系类型（比句级版更细：这里能区分方向，所以可以按动作语义细化）
VERB_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("hostile", ("骂", "打", "杀", "恨", "斥", "吵", "威胁", "报复", "攻击", "砍", "刺", "怒")),
    ("affection", ("爱", "喜欢", "思念", "吻", "抱", "牵", "心疼")),
    ("hierarchical", ("命令", "汇报", "请示", "吩咐", "指派", "安排", "任命", "要求")),
    ("collaborative", ("合作", "协助", "配合", "并肩")),
    ("dialogic", ("说", "道", "问", "答", "喊", "叫", "告诉", "解释", "劝", "笑", "叹")),
    ("social", ("见", "遇见", "认识", "介绍", "邀请", "拜访")),
]

# 常见动词的补充分支（jieba 词性给 v/vn，但词典命中优先）
_ALL_VERB_KWS = [(t, kws) for t, kws in VERB_TYPE_RULES]

WORD_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·]+")


def _ensure_dict(names: list[str]) -> None:
    """把人物词典灌进 jieba，保证规范名被切成整词而不是拆碎。"""
    for n in names:
        if n and len(n) >= 2:
            jieba.add_word(n, freq=10 ** 6, tag="nr")


def classify_verb(verb: str) -> str:
    """动词 → 关系类型；认不出返回 association。"""
    for rel_type, kws in _ALL_VERB_KWS:
        if any(k in verb for k in kws):
            return rel_type
    return DEFAULT_TYPE


def _verb_type_between(sentence: str) -> str:
    """句中没有明确匹配词时，退回句级谓词规则。"""
    for rel_type, kws in PREDICATE_RULES:
        if any(k in sentence for k in kws):
            return rel_type
    return DEFAULT_TYPE


def _tokens(sentence: str):
    return [(w, f) for w, f in pseg.cut(sentence) if w.strip() and WORD_RE.fullmatch(w)]


def extract_directed_relations(
    text: str, characters: list[str], min_cooccur: int = 1
) -> list[dict]:
    """抽取有向关系。

    规则（按优先级）：
    1. 同一句里出现两个及以上人物时，取"人物 动词 人物"的语序定方向：
       第一个出现的当主体、最后一个当客体（中间必须有动词）。
    2. 类型取中间动词；没有动词则用句级谓词，再没有就是 association。
    3. 三个人物以上时，按语序相邻配对（A→B、B→C），避免 N² 爆炸。

    Args:
        text: 全文
        characters: 规范名列表（已合并别名）
        min_cooccur: 同一 (主体,客体,类型) 至少出现几次才保留

    Returns:
        [{source, target, type, weight, evidence}]，按 weight 降序
    """
    names = sorted({c for c in characters if c}, key=len, reverse=True)
    _ensure_dict(names)

    bucket: dict[tuple[str, str, str], dict] = {}

    for sentence in split_sentences(text):
        toks = _tokens(sentence)
        if not toks:
            continue

        # 标出每个 token 是否是人名，并记录位置
        person_pos: list[int] = []
        for idx, (w, _f) in enumerate(toks):
            if w in names:
                person_pos.append(idx)
        if len(person_pos) < 2:
            continue

        # 相邻人物配对
        for a_idx, b_idx in zip(person_pos, person_pos[1:]):
            subj, obj = toks[a_idx][0], toks[b_idx][0]
            if subj == obj:
                continue

            # 动词可能在两位人物之间（"楚子航骂路明非"），也可能在第二个
            # 人物之后（"路明非对楚子航说"——中文里"说"常跟在宾语后面）。
            before = toks[a_idx + 1: b_idx]
            after = toks[b_idx + 1: b_idx + 4]
            verb = next((w for w, f in before if f.startswith("v")), "")
            if not verb:
                verb = next((w for w, f in after if f.startswith("v")), "")
            span = "".join(w for w, _f in before) or "".join(w for w, _f in after)
            if verb:
                rel_type = classify_verb(verb)
                if rel_type == DEFAULT_TYPE:
                    rel_type = _verb_type_between(span) if span else DEFAULT_TYPE
            else:
                rel_type = _verb_type_between(span) if span else DEFAULT_TYPE

            key = (subj, obj, rel_type)
            item = bucket.get(key)
            if item is None:
                bucket[key] = {
                    "source": subj,
                    "target": obj,
                    "type": rel_type,
                    "weight": 1,
                    "evidence": sentence[:120],
                }
            else:
                item["weight"] += 1

    relations = [r for r in bucket.values() if r["weight"] >= min_cooccur]
    relations.sort(key=lambda r: (-r["weight"], r["source"], r["target"]))
    return relations


def to_undirected(relations: list[dict]) -> list[dict]:
    """把有向边合并成无向边（兼容只吃无向图的调用方/前端）。"""
    merged: dict[tuple[str, str], dict] = {}
    for r in relations:
        a, b = sorted((r["source"], r["target"]))
        key = (a, b)
        item = merged.get(key)
        if item is None:
            merged[key] = {
                "source": a,
                "target": b,
                "type": r["type"],
                "weight": r["weight"],
                "evidence": r["evidence"],
                "directions": [{"from": r["source"], "to": r["target"], "type": r["type"]}],
            }
        else:
            item["weight"] += r["weight"]
            item["directions"].append(
                {"from": r["source"], "to": r["target"], "type": r["type"]}
            )
            if item["type"] == DEFAULT_TYPE and r["type"] != DEFAULT_TYPE:
                item["type"] = r["type"]
                item["evidence"] = r["evidence"]
    out = list(merged.values())
    out.sort(key=lambda r: -r["weight"])
    return out
