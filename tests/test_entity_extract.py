"""自动抽人物的过滤规则测试（遮天实测暴露的四类误判）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.base.entity_extract import pick_characters  # noqa: E402


def st(mentions, speech=0, action=0, non_person=0):
    return {"mentions": mentions, "speech": speech, "action": action, "non_person": non_person}


def test_frequent_speaker_is_kept():
    stats = {"叶凡": st(7000, speech=600, action=300)}
    assert pick_characters(stats, min_mentions=30, min_speech=3) == ["叶凡"]


def test_rare_name_dropped():
    stats = {"路人甲": st(5, speech=1)}
    assert pick_characters(stats, min_mentions=30) == []


def test_no_speech_no_action_dropped():
    stats = {"小镇": st(200, speech=0, action=0)}
    assert pick_characters(stats, min_mentions=30, min_speech=3) == []


def test_multi_tag_word_dropped():
    """「冷笑」在别处会被标成动词，多标签词判死。"""
    stats = {"冷笑": st(173, speech=70, action=4, non_person=40)}
    assert pick_characters(stats, min_mentions=30, min_speech=3) == []


def test_clan_or_skill_suffix_dropped():
    stats = {"姬家": st(400, speech=10), "西皇经": st(300, speech=10)}
    assert pick_characters(stats, min_mentions=30, min_speech=3) == []


def test_fragment_name_merged_into_longer():
    """「姬紫」是「姬紫月」的碎片，只保留长名。"""
    stats = {"姬紫月": st(112, speech=51), "姬紫": st(60, speech=20)}
    assert pick_characters(stats, min_mentions=30, min_speech=3) == ["姬紫月"]


def test_extra_ban_and_keep_win():
    stats = {"哪吒": st(100, speech=20), "黑皇": st(216, speech=32)}
    out = pick_characters(stats, min_mentions=30, min_speech=3, extra_ban={"哪吒"}, extra_keep={"黑皇"})
    assert out == ["黑皇"]


def test_sorted_by_mentions():
    stats = {"甲某": st(100, speech=10), "乙某": st(300, speech=10)}
    assert pick_characters(stats, min_mentions=30, min_speech=3)[0] == "乙某"
