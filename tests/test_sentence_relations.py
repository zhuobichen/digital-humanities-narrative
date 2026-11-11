"""句子级关系抽取的回归测试（P1-b）。

原来的滑动窗口共现会把所有人两两连上，85% 的边都是 association。
现在要求：同句出现才连边、类型由谓词推断、证据是原句。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network.sentence_relations import (  # noqa: E402
    detect_relation_type,
    extract_sentence_relations,
    split_sentences,
)


def test_split_sentences_by_chinese_punctuation():
    out = split_sentences("宝玉道。黛玉哭！宝钗笑？")
    assert out == ["宝玉道", "黛玉哭", "宝钗笑"]


def test_type_from_predicate():
    assert detect_relation_type("他骂了她") == "hostile"
    assert detect_relation_type("她爱他") == "affection"
    assert detect_relation_type("路明非说了一句话") == "dialogic"
    assert detect_relation_type("他们一起出发") == "collaborative"
    assert detect_relation_type("天气很好") == "association"


def test_only_same_sentence_pairs_get_edges():
    text = "路明非看着窗外。楚子航走了进来。"
    rels = extract_sentence_relations(text, ["路明非", "楚子航"])
    assert rels == []  # 不同句子不连边，这是与旧滑动窗口的关键区别


def test_same_sentence_creates_edge_with_evidence():
    text = "路明非对楚子航说：“走吧。”"
    rels = extract_sentence_relations(text, ["路明非", "楚子航"])
    assert len(rels) == 1
    edge = rels[0]
    assert {edge["source"], edge["target"]} == {"路明非", "楚子航"}
    assert edge["type"] == "dialogic"
    assert "楚子航" in edge["evidence"]


def test_weight_counts_cooccurring_sentences():
    text = "路明非和楚子航一起走。楚子航和路明非一起吃饭。楚子航独自发呆。"
    rels = extract_sentence_relations(text, ["路明非", "楚子航"])
    assert rels[0]["weight"] == 2


def test_min_cooccur_filters_weak_edges():
    text = "路明非和楚子航一起走。楚子航和恺撒点头。"
    all_edges = extract_sentence_relations(text, ["路明非", "楚子航", "恺撒"])
    strong = extract_sentence_relations(text, ["路明非", "楚子航", "恺撒"], min_cooccur=2)
    assert len(all_edges) == 2
    assert strong == []


def test_specific_type_wins_over_association():
    text = "路明非和楚子航站在门口。路明非和楚子航互相骂了起来。"
    rels = extract_sentence_relations(text, ["路明非", "楚子航"])
    assert rels[0]["type"] == "hostile"


def test_no_duplicate_edges_per_pair():
    text = "路明非和楚子航走。路明非和楚子航吃。路明非和楚子航睡。"
    rels = extract_sentence_relations(text, ["路明非", "楚子航"])
    assert len(rels) == 1
