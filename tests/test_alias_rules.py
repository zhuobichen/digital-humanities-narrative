"""别名合并与非人物过滤的回归测试。

场景来自《龙族1·火之晨曦》前 6 章的真实输出：合并前同一角色出现
（诺诺/陈墨瞳）、（昂热/校长/昂热校长）等多份，图谱里还混进了"卡塞尔"这类地名。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.base.alias_rules import (  # noqa: E402
    build_explicit_mapping,
    detect_alias_conflicts,
    merge_into_canonical,
)
from src.base.entity_filter import filter_persons, is_person  # noqa: E402


def test_merge_alias_into_canonical():
    mapping = {"陈墨瞳": ["诺诺"], "路明非": ["明非"]}
    merged = merge_into_canonical(["诺诺", "陈墨瞳", "路明非", "楚子航"], mapping)
    assert merged == ["陈墨瞳", "路明非", "楚子航"]


def test_merge_keeps_order_of_first_mention():
    mapping = {"陈墨瞳": ["诺诺"]}
    assert merge_into_canonical(["诺诺", "楚子航", "陈墨瞳"], mapping) == ["陈墨瞳", "楚子航"]


def test_build_mapping_drops_self_and_blanks():
    result = build_explicit_mapping({"路明非": ["路明非", " ", "明非"]})
    assert result == {"路明非": ["明非"]}


def test_alias_pointing_to_two_people_is_rejected():
    with pytest.raises(ValueError):
        build_explicit_mapping({"甲": ["X"], "乙": ["X"]})


def test_detect_conflicts_lists_both_owners():
    conflicts = detect_alias_conflicts({"甲": ["X"], "乙": ["X"], "丙": ["Y"]})
    assert conflicts == {"X": ["甲", "乙"]}


def test_non_person_filtered_from_characters():
    # 卡塞尔是学院名：词典里混着它，但不应成为人物节点
    assert not is_person("卡塞尔", "longzu")
    assert filter_persons(["路明非", "卡塞尔", "楚子航"], "longzu") == ["路明非", "楚子航"]


def test_unknown_book_keeps_generic_filter_only():
    assert filter_persons(["张三", "学校"], "unknown-book") == ["张三"]


def test_extra_ban_list_is_respected():
    assert filter_persons(["张三", "李四"], "", extra={"李四"}) == ["张三"]
