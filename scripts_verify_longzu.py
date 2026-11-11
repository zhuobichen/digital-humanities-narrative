"""《龙族》回归：别名合并 + 非人物过滤 修完之后，网络到底变成了什么样。"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.base.alias_rules import build_explicit_mapping, detect_alias_conflicts
from src.base.character_dict import LONGZU_CHARACTERS, LONGZU_MAPPING
from src.pipeline import NarrativePipeline

TXT = Path("data/龙族/龙族1_前6章.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)


def run(use_mapping: bool):
    text = TXT.read_text("utf-8")
    mapping = build_explicit_mapping(LONGZU_MAPPING) if use_mapping else None
    t0 = time.time()
    pipe = NarrativePipeline(
        strict_dict_mode=True,
        character_names=LONGZU_CHARACTERS,
        explicit_mapping=mapping,
    )
    r = pipe.run([text], book="longzu", verbose=False)
    elapsed = time.time() - t0

    types = Counter(x.type for x in r.relations)
    evi = sum(1 for x in r.relations if (x.context_snippet or "").strip())
    return {
        "mapping": use_mapping,
        "elapsed_s": round(elapsed, 2),
        "n_characters": len(r.characters),
        "characters": r.characters,
        "n_relations": len(r.relations),
        "relation_types": dict(types),
        "evidence_ratio": round(evi / max(1, len(r.relations)), 3),
        "density": round(r.network_analysis.density, 4),
        "main_character": r.network_analysis.main_character,
        "events": len(r.events),
        "distinct_summaries": len({p.personality_summary for p in r.profiles}),
        "roles": dict(Counter(p.role_in_story for p in r.profiles)),
    }


def main():
    conflicts = detect_alias_conflicts(LONGZU_MAPPING)
    print(f"映射表冲突：{conflicts or '无'}")
    print(f"映射表规模：{len(LONGZU_MAPPING)} 个规范名，"
          f"{sum(len(v) for v in LONGZU_MAPPING.values())} 个别名，"
          f"总候选 {len(LONGZU_CHARACTERS)}")

    out = {}
    for label, flag in (("合并前（只有词典，无映射）", False), ("合并后（启用别名映射+过滤）", True)):
        res = run(flag)
        out["before" if not flag else "after"] = res
        print(f"\n=== {label} ===")
        print(f"  人物 {res['n_characters']} 个: {res['characters']}")
        print(f"  关系 {res['n_relations']} 条  类型 {res['relation_types']}")
        print(f"  密度 {res['density']}  主角 {res['main_character']}  证据覆盖 {res['evidence_ratio']:.0%}")
        print(f"  画像去重 {res['distinct_summaries']} 种 / {res['n_characters']} 人  角色 {res['roles']}")

    (OUT / "longzu_after.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print("\n已写出 out/longzu_after.json")


if __name__ == "__main__":
    main()
