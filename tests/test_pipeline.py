"""端到端测试：验证完整流水线"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base.ner import NamedEntityRecognizer
from src.base.alias import CharacterAliasResolver
from src.network.relations import RelationshipExtractor
from src.network.graph import CharacterNetworkBuilder
from src.network.visualize import NetworkVisualizer
from src.plot.timeline import TimelineExtractor
from src.plot.causal import CausalChainAnalyzer
from src.character.profiler import CharacterProfiler
from src.character.dynamics import RelationshipDynamicsAnalyzer
from src.narrative.destiny import DestinyPredictor
from src.narrative.generator import NarrativeGenerator
from src.pipeline import NarrativePipeline


# ---- 测试文本 ----
TEST_NAMES = ["张三", "李四", "王经理"]

TEST_TEXTS = [
    "2024年1月15日，张三和李四启动了新项目。",
    "昨天，王经理审查了项目进度，对进展表示满意。",
    "李四和王经理讨论了技术方案，张三也参加了会议。",
    "张三向王经理汇报了工作进展，李四提出了改进建议。",
    "今天，张三和李四在餐厅一起吃饭，讨论了下周的计划。",
    "王经理决定让张三负责新模块的开发。",
    "李四对王经理的决定有些不满，认为应该由自己负责。",
    "上周，张三完成了第一阶段的开发工作。",
    "今天上午，王经理表扬了张三的工作成果。",
    "李四经过思考后，理解了王经理的决定，开始配合张三的工作。",
    "下周，团队将向客户展示项目成果。",
    "张三和李四一起准备了Demo演示。",
]


def test_ner():
    """Phase 1: 测试 NER（词典模式，不依赖 spaCy 模型）"""
    ner = NamedEntityRecognizer(
        strict_dict_mode=True, character_names=TEST_NAMES
    )
    chars = ner.extract_characters("\n".join(TEST_TEXTS))
    assert len(chars) > 0, "NER 应识别到人物"
    assert "张三" in chars, "应识别到张三"
    print(f"  [PASS] NER: 识别到 {len(chars)} 个实体: {chars}")
    return chars


def test_alias(raw_characters):
    """Phase 1: 测试别名识别"""
    resolver = CharacterAliasResolver()
    groups = resolver.resolve_aliases("\n".join(TEST_TEXTS), raw_characters)
    print(f"  [PASS] Alias: 发现 {len(groups)} 个别名组")
    for g in groups:
        print(f"    {g['canonical_name']} → {g['aliases']}")
    return groups


def test_relations(characters):
    """Phase 2: 测试关系抽取"""
    extractor = RelationshipExtractor()
    relations = extractor.extract_relations_with_sentiment(
        "\n".join(TEST_TEXTS), characters
    )
    relations = [r.model_dump() if hasattr(r, "model_dump") else r for r in relations]
    assert len(relations) > 0, "应抽取到关系"
    print(f"  [PASS] Relations: 抽取 {len(relations)} 条关系")
    for r in relations[:3]:
        print(f"    {r['source']} - {r['target']}: {r['type']}")
    return relations


def test_network(rel_objs):
    """Phase 2: 测试网络构建"""
    builder = CharacterNetworkBuilder()
    G = builder.build_network(rel_objs)
    analysis = builder.analyze_network()
    analysis = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis
    assert analysis["num_characters"] > 0
    assert analysis["main_character"] is not None
    print(f"  [PASS] Network: {analysis['num_characters']} 节点, "
          f"{analysis['num_relations']} 边, 主角={analysis['main_character']}")
    return builder


def test_timeline(characters):
    """Phase 3: 测试时间线"""
    extractor = TimelineExtractor()
    events = extractor.extract_events(TEST_TEXTS, characters)
    events = [e.model_dump() if hasattr(e, "model_dump") else e for e in events]
    assert len(events) > 0
    timed = [e for e in events if e.get("time")]
    print(f"  [PASS] Timeline: {len(events)} 事件, {len(timed)} 有时间信息")
    return events


def test_causal(event_objs):
    """Phase 3: 测试因果分析"""
    analyzer = CausalChainAnalyzer()
    chains = analyzer.analyze_causal_chains(event_objs)
    chains = [c.model_dump() if hasattr(c, "model_dump") else c for c in chains]
    arc = analyzer.build_narrative_arc(event_objs)
    arc = arc.model_dump() if hasattr(arc, "model_dump") else arc
    assert len(chains) == len(event_objs)
    total_in_arc = sum(len(v) for v in arc.values())
    assert total_in_arc == len(event_objs)
    print(f"  [PASS] Causal: {len(chains)} 因果链, "
          f"{sum(1 for c in chains if c['is_turning_point'])} 个转折点")
    return chains, arc


def test_profiler(characters):
    """Phase 4: 测试人物刻画"""
    profiler = CharacterProfiler()
    profiles = profiler.profile_all(characters, TEST_TEXTS)
    profiles = [p.model_dump() if hasattr(p, "model_dump") else p for p in profiles]
    assert len(profiles) > 0
    for p in profiles:
        assert "traits" in p
        assert "role_in_story" in p
    print(f"  [PASS] Profiler: {len(profiles)} 个人物画像")
    for p in profiles:
        print(f"    {p['name']}: {p['role_in_story']} - {p['personality_summary'][:60]}")
    return profiles


def test_dynamics(rel_objs):
    """Phase 4: 测试关系动态"""
    analyzer = RelationshipDynamicsAnalyzer()
    evolution = analyzer.analyze_from_relations(rel_objs)
    assert len(evolution) > 0
    print(f"  [PASS] Dynamics: {len(evolution)} 组关系动态")
    return evolution


def test_destiny(profile_objs, arc):
    """Phase 5: 测试命运预测"""
    from src.models import NarrativeArc
    predictor = DestinyPredictor()
    arc_obj = arc if not isinstance(arc, dict) else NarrativeArc(**arc)
    for p in profile_objs:
        result = predictor.predict_destiny(p, arc_obj)
        # 模块返回 DestinyPrediction，旧断言当 dict 用
        assert hasattr(result, "character") or "character" in result
        assert hasattr(result, "overall_outlook") or "overall_outlook" in result
    print(f"  [PASS] Destiny: {len(profile_objs)} 个人物命运预测完成")
    return True


def test_generator(results_dict):
    """Phase 5: 测试故事生成"""
    generator = NarrativeGenerator()
    summary = generator.generate_summary(results_dict)
    assert "title" in summary
    story = generator.generate_story(results_dict, style="dramatic")
    assert len(story) > 0

    # 人物小传
    if results_dict["profiles"]:
        bio = generator.generate_character_bio(results_dict["profiles"][0])
        assert len(bio) > 0

    print(f"  [PASS] Generator: 标题={summary['title']}, 故事长度={len(story)} 字符")
    return summary, story


def test_pipeline():
    """端到端流水线测试"""
    print("\n  --- 端到端流水线 ---")
    pipeline = NarrativePipeline(strict_dict_mode=True, character_names=TEST_NAMES)
    results = pipeline.run(TEST_TEXTS, verbose=False)
    results = results.model_dump() if hasattr(results, "model_dump") else results

    assert len(results["characters"]) > 0
    assert len(results["relations"]) > 0
    assert len(results["events"]) > 0
    assert len(results["profiles"]) > 0
    assert "story" in results

    # 快速分析
    quick = pipeline.quick_analysis(TEST_TEXTS)
    quick = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in quick.items()}
    assert quick["num_characters"] > 0
    assert quick["main_character"] is not None

    print(f"  [PASS] Pipeline: {quick['num_characters']} 人物, "
          f"主角={quick['main_character']}, {quick['num_events']} 事件")
    return results


def run_all():
    print("=" * 60)
    print("  数字人文叙事分析 - 测试套件")
    print("=" * 60)

    # Phase 1
    print("\n[Phase 1] 基础提取")
    chars = test_ner()
    test_alias(chars)

    # Phase 2
    print("\n[Phase 2] 关系网络")
    relations = test_relations(chars)
    builder = test_network(relations)

    # Phase 3
    print("\n[Phase 3] 情节重构")
    events = test_timeline(chars)
    chains, arc = test_causal(events)

    # Phase 4
    print("\n[Phase 4] 人物刻画")
    profiles = test_profiler(chars)
    test_dynamics(relations)

    # Phase 5
    print("\n[Phase 5] 叙事生成")
    test_destiny(profiles, arc)

    # 构建综合结果给 generator
    results_dict = {
        "network_analysis": builder.analyze_network(),
        "profiles": profiles,
        "events": events,
        "causal_chains": chains,
        "narrative_arc": arc,
    }
    test_generator(results_dict)

    # 端到端
    test_pipeline()

    print("\n" + "=" * 60)
    print("  所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
