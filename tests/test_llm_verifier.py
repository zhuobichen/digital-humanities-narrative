"""LLM 复核的单元测试。

没有 API key 时必须"安静跳过"，有 key 时按模型判定剔除/纠正。
这里用假 extractor 模拟模型返回，不真的联网。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.network.llm_verifier import (  # noqa: E402
    review_candidates,
    summarize_review,
)

CANDIDATES = [
    {"source": "路明非", "target": "楚子航", "type": "hostile", "weight": 3, "evidence": "路明非骂了楚子航"},
    {"source": "陈雯雯", "target": "路明非", "type": "association", "weight": 2, "evidence": "陈雯雯和路明非"},
    {"source": "诺诺", "target": "路明非", "type": "dialogic", "weight": 1, "evidence": "诺诺说"},
]


class FakeExtractor:
    def __init__(self, replies, available=True):
        self.replies = list(replies)
        self.available = available
        self.calls = 0

    def _call_llm_with_retry(self, prompt):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_no_api_key_returns_candidates_untouched():
    fake = FakeExtractor([], available=False)
    out = review_candidates(fake, CANDIDATES)
    assert len(out) == len(CANDIDATES)
    assert all(c["reviewed"] is False for c in out)
    assert fake.calls == 0  # 不允许在没 key 时发请求


def test_invalid_edge_is_dropped():
    fake = FakeExtractor([
        '{"valid": false, "type": "hostile", "reason": "证据句没有冲突"}',
        '{"valid": true, "type": "association", "reason": ""}',
        '{"valid": true, "type": "dialogic", "reason": ""}',
    ])
    out = review_candidates(fake, CANDIDATES)
    assert len(out) == 2
    assert all(c["reviewed"] for c in out)


def test_type_is_corrected():
    fake = FakeExtractor([
        '{"valid": true, "type": "social", "reason": "只是见面"}',
        '{"valid": true, "type": "association", "reason": ""}',
        '{"valid": true, "type": "dialogic", "reason": ""}',
    ])
    out = review_candidates(fake, CANDIDATES)
    assert out[0]["type"] == "social"


def test_unknown_type_keeps_original():
    fake = FakeExtractor([
        '{"valid": true, "type": "totally_made_up", "reason": ""}',
        '{"valid": true, "type": "association", "reason": ""}',
        '{"valid": true, "type": "dialogic", "reason": ""}',
    ])
    out = review_candidates(fake, CANDIDATES)
    assert out[0]["type"] == "hostile"


def test_failure_falls_back_to_rule_result():
    fake = FakeExtractor([
        RuntimeError("boom"),
        '{"valid": true, "type": "association", "reason": ""}',
        '{"valid": true, "type": "dialogic", "reason": ""}',
    ])
    out = review_candidates(fake, CANDIDATES)
    assert len(out) == 3
    assert out[0]["reviewed"] is False
    assert "RuntimeError" in out[0]["reason"]


def test_bad_json_is_tolerated():
    fake = FakeExtractor(["not json", '{"valid": true}', '{"valid": true}'])
    out = review_candidates(fake, CANDIDATES)
    assert len(out) == 3
    assert out[0]["reviewed"] is False


def test_max_items_limits_llm_calls():
    fake = FakeExtractor(['{"valid": true, "type": "social", "reason": ""}'])
    out = review_candidates(fake, CANDIDATES, max_items=1)
    assert fake.calls == 1
    assert len(out) == 3
    assert sum(1 for c in out if c["reviewed"]) == 1


def test_summary_counts():
    before = CANDIDATES
    after = [
        {**CANDIDATES[0], "reviewed": True, "type": "social"},
        {**CANDIDATES[1], "reviewed": False},
    ]
    s = summarize_review(before, after)
    assert s["candidates"] == 3
    assert s["kept"] == 2
    assert s["dropped"] == 1
    assert s["reviewed"] == 1
    assert s["types_after"]["social"] == 1
