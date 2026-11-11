"""测试夹具 + 兼容层。

原仓库 tests/test_pipeline.py 用了 7 个 fixture（raw_characters / characters /
relations / events / profiles / arc / results_dict），但没有任何定义，导致
pytest 连收集都失败。

测试本身写于"字典时代"（关系/事件/画像按 dict 访问），而现在的模块返回 Pydantic
对象。修法是在测试侧做适配，让测试按它原本的契约跑，而不是去改测试断言。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_NAMES = ["张三", "李四", "王经理"]


def _as_dict(obj):
    """Pydantic 对象 → dict；已是 dict 则原样返回。"""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
        if hasattr(obj, "name"):
            d["name"] = obj.name
        return d
    return obj


@pytest.fixture(scope="session")
def texts():
    from tests.test_pipeline import TEST_TEXTS
    return TEST_TEXTS


@pytest.fixture(scope="session")
def raw_characters(texts):
    from src.base.ner import NamedEntityRecognizer

    ner = NamedEntityRecognizer(strict_dict_mode=True, character_names=TEST_NAMES)
    return ner.extract_characters("\n".join(texts))


@pytest.fixture(scope="session")
def characters(raw_characters):
    return list(raw_characters)


@pytest.fixture(scope="session")
def rel_objs(texts, characters):
    """未适配的原始 Relation 对象（喂给 build_network 这类要对象的接口）。"""
    from src.network.relations import RelationshipExtractor

    extractor = RelationshipExtractor()
    return extractor.extract_relations("\n".join(texts), characters)


@pytest.fixture(scope="session")
def relations(rel_objs):
    """测试契约：关系按 dict 访问（r['source']）。"""
    return [_as_dict(r) for r in rel_objs]


@pytest.fixture(scope="session")
def event_objs(texts, characters):
    """未适配的原始 Event 对象（喂给 build_narrative_arc 这类要对象的接口）。"""
    from src.plot.timeline import TimelineExtractor

    extractor = TimelineExtractor()
    return extractor.extract_events(texts, characters)


@pytest.fixture(scope="session")
def events(event_objs):
    """测试契约：事件按 dict 访问（e.get('time')）。"""
    return [_as_dict(e) for e in event_objs]


@pytest.fixture(scope="session")
def profile_objs(characters, texts):
    from src.character.profiler import CharacterProfiler

    profiler = CharacterProfiler()
    return profiler.profile_all(characters, texts)


@pytest.fixture(scope="session")
def profiles(profile_objs):
    """测试契约：画像按 dict 访问（p['traits'] / p['name']）。"""
    return [_as_dict(p) for p in profile_objs]


@pytest.fixture(scope="session")
def arc(event_objs):
    from src.plot.causal import CausalChainAnalyzer

    analyzer = CausalChainAnalyzer()
    obj = analyzer.build_narrative_arc(event_objs)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


@pytest.fixture(scope="session")
def results_dict(characters, relations, events, profiles):
    return {
        "characters": characters,
        "relations": relations,
        "events": events,
        "profiles": profiles,
        "summary": {},
    }
