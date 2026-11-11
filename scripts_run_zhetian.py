"""《遮天》实测：自动抽人物 + 人工纠正表，量化"垃圾节点"对关系图的影响。

用法：python scripts_run_zhetian.py
产出：out/zhetian_compare.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.base.entity_extract import collect_stats, pick_characters
from src.pipeline import NarrativePipeline

SRC = Path("data/遮天/遮天_utf8.txt")
OUT = Path("out")
OUT.mkdir(exist_ok=True)
MAX_CHARS = 1_200_000

# 自动抽取后仍混进来的高频非人物词（人工过一眼正文就能确认）
ZT_BAN = {
    "冷笑", "闻言", "巴掌", "光辉", "生辉", "小镇", "玄奥", "森森", "瑞彩", "宝杵",
    "向前走", "师兄", "老伯", "尼姑", "长青", "文昌", "怀远", "大胡子", "古怪",
    "古老", "金光", "光芒", "众人", "姬家", "姜家", "雷音寺", "金翅小鹏",
}


def load_text() -> str:
    return SRC.read_text("utf-8")[:MAX_CHARS]


def run(characters: list[str], text: str, label: str) -> dict:
    t0 = time.time()
    pipe = NarrativePipeline(strict_dict_mode=True, character_names=characters)
    r = pipe.run([text], book="遮天", verbose=False)
    elapsed = time.time() - t0
    types = Counter(x.type for x in r.relations)
    junk = set(characters) - set(characters)  # placeholder
    return {
        "label": label,
        "elapsed_s": round(elapsed, 1),
        "n_characters": len(r.characters),
        "characters": r.characters,
        "n_relations": len(r.relations),
        "relation_types": dict(types),
        "evidence_ratio": round(
            sum(1 for x in r.relations if (x.context_snippet or "").strip()) / max(1, len(r.relations)), 3
        ),
        "density": round(r.network_analysis.density, 4),
        "main_character": r.network_analysis.main_character,
        "role_dist": dict(Counter(p.role_in_story for p in r.profiles)),
        "top_relations": [
            {"source": x.source, "type": x.type, "target": x.target,
             "evidence": (x.context_snippet or "")[:45]}
            for x in r.relations[:12]
        ],
        "_junk": sorted(junk),
    }


def pollution(relations: list[dict], junk: set[str]) -> float:
    if not relations:
        return 0.0
    hit = sum(1 for r in relations if r["source"] in junk or r["target"] in junk)
    return hit / len(relations)


def main():
    text = load_text()
    print(f"[i] 样本 {len(text):,} 字")

    stats = collect_stats(text)
    auto = pick_characters(stats, book="遮天", min_mentions=30, min_speech=3)
    clean = [n for n in auto if n not in ZT_BAN]
    print(f"[i] 自动 {len(auto)} 人 → 清洗后 {len(clean)} 人")

    res_auto = run(auto, text, "auto")
    res_clean = run(clean, text, "cleaned")

    for res, junk in ((res_auto, ZT_BAN), (res_clean, set())):
        res["junk_edges"] = pollution(res["top_relations"], junk) if junk else 0.0

    print(f"\n{'口径':8s} {'人物':>4s} {'关系':>5s} {'垃圾边(抽样)':>10s} {'密度':>7s} {'主角':>6s}")
    for res, junk in ((res_auto, ZT_BAN), (res_clean, set())):
        j = pollution(res["top_relations"], junk) if junk else 0.0
        print(f"{res['label']:8s} {res['n_characters']:>4d} {res['n_relations']:>5d} "
              f"{j:>9.0%} {res['density']:>7.3f} {res['main_character']}")

    print("\n清洗后的边（前 10）：")
    for r in res_clean["top_relations"]:
        print(f"  {r['source']} →[{r['type']}]→ {r['target']}  «{r['evidence']}»")

    (OUT / "zhetian_compare.json").write_text(
        json.dumps({"auto": res_auto, "cleaned": res_clean, "ban": sorted(ZT_BAN)},
                   ensure_ascii=False, indent=1), "utf-8")
    print("\n已写出 out/zhetian_compare.json")


if __name__ == "__main__":
    main()
