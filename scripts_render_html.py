"""为已完成的语料生成自包含 HTML 关系图（零外部依赖，可直接双击打开）。

产出：
  out/龙族_前6章_人物关系.html
  out/遮天_前120万字_人物关系.html
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.base.alias_rules import build_explicit_mapping, merge_into_canonical
from src.base.character_dict import LONGZU_CHARACTERS, LONGZU_MAPPING
from src.pipeline import NarrativePipeline

OUT = Path("out")
OUT.mkdir(exist_ok=True)


def render_longzu() -> str:
    text = Path("data/龙族/龙族1_前6章.txt").read_text("utf-8")
    mapping = build_explicit_mapping(LONGZU_MAPPING)
    alias_map = {a: c for c, vs in mapping.items() for a in vs}
    for a, c in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        text = text.replace(a, c)
    pipe = NarrativePipeline(
        strict_dict_mode=True,
        character_names=LONGZU_CHARACTERS,
        explicit_mapping=mapping,
    )
    r = pipe.run([text], book="longzu", verbose=False)
    path = OUT / "龙族_前6章_人物关系.html"
    pipe.visualizer.set_graph(pipe.network_builder.G)
    pipe.visualizer.to_standalone_html(
        str(path), title="《龙族1·火之晨曦》前 6 章 · 人物关系"
    )
    print(f"[OK] {path}  {path.stat().st_size/1024:.0f} KB  "
          f"{len(r.characters)} 人 / {len(r.relations)} 边")
    return str(path)


def render_zhetian() -> str:
    from scripts_run_zhetian import ZT_BAN, load_text
    from src.base.entity_extract import collect_stats, pick_characters

    text = load_text()
    stats = collect_stats(text)
    names = [n for n in pick_characters(stats, book="遮天", min_mentions=30, min_speech=3)
             if n not in ZT_BAN]
    pipe = NarrativePipeline(strict_dict_mode=True, character_names=names)
    r = pipe.run([text], book="遮天", verbose=False)
    path = OUT / "遮天_前120万字_人物关系.html"
    pipe.visualizer.set_graph(pipe.network_builder.G)
    pipe.visualizer.to_standalone_html(
        str(path), title="《遮天》前 120 万字 · 人物关系（自动抽人物 + 人工清洗）"
    )
    print(f"[OK] {path}  {path.stat().st_size/1024:.0f} KB  "
          f"{len(r.characters)} 人 / {len(r.relations)} 边")
    return str(path)


if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "longzu"):
        render_longzu()
    if which in ("all", "zhetian"):
        render_zhetian()
    print(f"总耗时 {time.time()-t0:.1f}s")
