"""有向关系抽取的准确度验收：人工标定 + 龙族实测。

用法：
    python _eval_directed.py           # 标定集 + 龙族抽样
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.network.directed_relations import extract_directed_relations  # noqa: E402

# 人工标定：句子里"谁对谁做了什么"。期望 (主体, 类型, 客体)
GOLD = [
    ("路明非对楚子航说：走吧。", ["路明非", "楚子航"], ("路明非", "dialogic", "楚子航")),
    ("楚子航骂了路明非一顿。", ["楚子航", "路明非"], ("楚子航", "hostile", "路明非")),
    ("诺诺喜欢路明非。", ["诺诺", "路明非"], ("诺诺", "affection", "路明非")),
    ("路明非命令楚子航撤退。", ["路明非", "楚子航"], ("路明非", "hierarchical", "楚子航")),
    ("陈雯雯带路明非去了文学社。", ["陈雯雯", "路明非"], ("陈雯雯", "association", "路明非")),
    ("恺撒和路明非并肩作战。", ["恺撒", "路明非"], ("恺撒", "association", "路明非")),
    ("路明非大声喊诺诺的名字。", ["路明非", "诺诺"], ("路明非", "dialogic", "诺诺")),
    ("赵孟华看不起路明非。", ["赵孟华", "路明非"], ("赵孟华", "association", "路明非")),
]


def eval_gold():
    ok_dir = ok_type = 0
    detail = []
    for sentence, names, expect in GOLD:
        got = extract_directed_relations(sentence, names)
        top = max(
            got,
            key=lambda r: r["weight"],
            default=None,
        )
        if top is None:
            detail.append((sentence, "无结果", expect))
            continue
        dir_ok = (top["source"], top["target"]) == (expect[0], expect[2])
        type_ok = top["type"] == expect[1]
        ok_dir += dir_ok
        ok_type += type_ok
        detail.append((sentence, f"{top['source']}→[{top['type']}]→{top['target']}",
                       "OK" if dir_ok and type_ok else ("方向对/类型错" if dir_ok else "方向错")))
    n = len(GOLD)
    print(f"标定集 {n} 句：方向正确 {ok_dir}/{n} = {ok_dir / n:.0%}；类型正确 {ok_type}/{n} = {ok_type / n:.0%}")
    for s, got, verdict in detail:
        print(f"  [{verdict:8s}] {s}  ⇒ {got}")
    return ok_dir / n


def eval_longzu():
    from src.base.character_dict import LONGZU_CHARACTERS, LONGZU_MAPPING
    from src.base.alias_rules import build_explicit_mapping, merge_into_canonical

    text = Path("data/龙族/龙族1_前6章.txt").read_text("utf-8")
    mapping = build_explicit_mapping(LONGZU_MAPPING)
    alias_map = {a: c for c, vs in mapping.items() for a in vs}
    for a, c in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(a, c)
    chars = merge_into_canonical(LONGZU_CHARACTERS, mapping)
    rels = extract_directed_relations(text, chars)

    typed = [r for r in rels if r["type"] != "association"]
    print(f"\n龙族前 6 章：有向关系 {len(rels)} 条，其中带具体类型 {len(typed)} 条 "
          f"({len(typed) / max(1, len(rels)):.0%})")
    print("样本（前 12 条）：")
    for r in rels[:12]:
        print(f"  {r['source']:6s} →[{r['type']:13s}]→ {r['target']:6s}  w={r['weight']:3d}  «{r['evidence'][:36]}»")
    return rels


if __name__ == "__main__":
    eval_gold()
    eval_longzu()
