"""用《龙族1·火之晨曦》前 6 章跑完整流水线，并与《红楼梦》结果做对比。

重点观察：
1. 没有领域词典（HONGLOUMENG_CHARACTERS 是红楼梦专用）时，纯规则 NER 能不能认出主角
2. 关系数/关系类型/证据句的分布
3. 人物画像是否同质化（上次红楼梦 7 人全"主角/尽责认真"）
4. 耗时随文本量的变化
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.base.character_dict import HONGLOUMENG_CHARACTERS
from src.pipeline import NarrativePipeline

TXT = Path("data/龙族/龙族1_前6章.txt")
OUT = Path("out"); OUT.mkdir(exist_ok=True)

# 规则 NER 需要人名词典；红楼梦词典对龙族没用，这里给一个"最小种子表"，
# 只放读者一眼就知道的主角/关键角色，用来检验"给了一点先验之后能跑多好"。
LONGZU_SEED = [
    "路明非", "路鸣泽", "陈墨瞳", "诺诺", "楚子航", "恺撒", "凯撒",
    "芬格尔", "昂热", "康斯坦丁", "叶胜", "古德里安", "苏茜",
    "泽非", "白王", "黑王", "卡塞尔", "赵孟华", "陈雯雯",
]


def run(seed_mode: str):
    text = TXT.read_text("utf-8")
    names = LONGZU_SEED if seed_mode == "seed" else []
    t0 = time.time()
    pipe = NarrativePipeline(strict_dict_mode=True, character_names=names)
    r = pipe.run([text], book="龙族1·火之晨曦", verbose=False)
    elapsed = time.time() - t0

    rel_types = Counter(x.type for x in r.relations)
    with_evi = sum(1 for x in r.relations if (x.context_snippet or "").strip())
    roles = Counter(p.role_in_story for p in r.profiles)
    summaries = {p.personality_summary for p in r.profiles}
    return {
        "seed_mode": seed_mode,
        "chars": len(text),
        "elapsed_s": round(elapsed, 2),
        "characters": r.characters,
        "n_relations": len(r.relations),
        "relation_types": dict(rel_types),
        "evidence_ratio": round(with_evi / max(1, len(r.relations)), 3),
        "events": len(r.events),
        "profiles": [
            {"name": p.name, "role": p.role_in_story, "mentions": p.total_mentions,
             "presence": round(p.presence_ratio, 4), "summary": p.personality_summary[:50]}
            for p in r.profiles
        ],
        "role_dist": dict(roles),
        "distinct_summaries": len(summaries),
        "network": {
            "nodes": r.network_analysis.num_characters if hasattr(r.network_analysis, "num_characters") else None,
            "edges": r.network_analysis.num_relations if hasattr(r.network_analysis, "num_relations") else None,
            "main": r.network_analysis.main_character,
            "density": round(r.network_analysis.density, 4),
        },
        "story_chars": len(r.story or ""),
    }


def main():
    results = {}
    for mode in ("seed", "no_seed"):
        print(f"\n=== 运行 {mode} ===")
        res = run(mode)
        results[mode] = res
        print(f"耗时 {res['elapsed_s']}s  人物 {len(res['characters'])}  关系 {res['n_relations']}  "
              f"证据覆盖 {res['evidence_ratio']:.0%}  事件 {res['events']}")
        print(f"人物: {res['characters']}")
        print(f"关系类型: {res['relation_types']}")
        print(f"画像总结去重后 {res['distinct_summaries']} 种 / {len(res['profiles'])} 人；角色分布 {res['role_dist']}")
        print(f"网络: {res['network']}")
    (OUT / "longzu_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    print("\n已写出 out/longzu_report.json")


if __name__ == "__main__":
    main()
