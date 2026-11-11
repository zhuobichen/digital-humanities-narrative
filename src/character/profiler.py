"""人物性格刻画器

基于语言使用模式分析人物的性格特质、情感倾向和沟通风格。
"""

import json
from typing import Any, List, Optional, Dict
from collections import Counter

from ..interfaces import BaseCharacterProfiler
from ..models import CharacterProfile, Traits, EmotionProfile, CommunicationStyle, Relation

class LLMCharacterProfiler(BaseCharacterProfiler):
    """基于大模型的人物性格刻画器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.role_thresholds = {
            "protagonist": 0.3,
            "supporting": 0.2,
            "minor": 0.1,
            "background": 0.0,
        }

    def profile_all(
        self, characters: list[str], texts: list[str]
    ) -> List[CharacterProfile]:
        """批量刻画所有人物"""
        profiles = []
        # 只刻画出现频率最高的前 10 个人物，节省 API 调用时间
        char_counts = {c: sum(1 for t in texts if c in t) for c in characters}
        top_chars = sorted(char_counts.keys(), key=lambda x: char_counts[x], reverse=True)[:10]
        
        for name in characters:
            if name in top_chars and char_counts[name] > 0:
                profiles.append(self.profile_character(name, texts, characters))
            else:
                # 简单回退给不重要的人物
                role = self._infer_role(name, [t for t in texts if name in t], texts, characters)
                profiles.append(CharacterProfile(
                    name=name,
                    role_in_story=role,
                    total_mentions=char_counts[name],
                    # presence_ratio 是"在多少段文本里出现过"的占比，不是字数占比。
                    # 原来用 char_counts / len(texts)，而 texts 只是"段数"（长文往往 1 段），
                    # 导致任何出现过的角色都被算成 1.0。
                    presence_ratio=sum(1 for t in texts if name in t) / max(len(texts), 1),
                    traits=Traits(),
                    emotions=EmotionProfile(),
                    communication_style=CommunicationStyle(),
                    personality_summary="出现次数较少，暂无足够信息。"
                ))
        return profiles

    def profile_character(
        self, name: str, texts: list[str], all_characters: Optional[list[str]] = None
    ) -> CharacterProfile:
        character_texts = [t for t in texts if name in t]
        presence_ratio = len(character_texts) / max(len(texts), 1)
        role = self._infer_role(name, character_texts, texts, all_characters or [])
        
        # 将文本拼接，限制长度
        context = "\n".join(character_texts)[:3000]
        
        prompt = f"""
你是一个古典文学数字人文分析专家。请根据以下关于人物【{name}】的文本片段，分析其性格特质和情感倾向。

请输出 JSON 格式，包含以下结构:
{{
  "traits": {{
    "leadership": 0.0到1.0的浮点数,
    "creativity": 0.0到1.0的浮点数,
    "agreeableness": 0.0到1.0的浮点数,
    "conscientiousness": 0.0到1.0的浮点数,
    "extraversion": 0.0到1.0的浮点数,
    "emotional_stability": 0.0到1.0的浮点数,
    "assertiveness": 0.0到1.0的浮点数,
    "cooperativeness": 0.0到1.0的浮点数
  }},
  "emotions": {{
    "dominant": "主导情感(例如: 愤怒, 喜悦, 悲伤, 中立等)",
    "valence": "情感效价(positive, negative, 或 neutral)"
  }},
  "personality_summary": "一段简短的人物性格与行为总结（50字以内）"
}}

文本片段:
{context}
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            data = json.loads(response.choices[0].message.content)
            
            traits_data = data.get("traits", {})
            emotions_data = data.get("emotions", {})
            
            profile = CharacterProfile(
                name=name,
                role_in_story=role,
                total_mentions=len(character_texts),
                presence_ratio=presence_ratio,
                traits=Traits(**{k: float(v) for k, v in traits_data.items() if hasattr(Traits, k)}),
                emotions=EmotionProfile(
                    dominant=emotions_data.get("dominant", "neutral"),
                    valence=emotions_data.get("valence", "neutral"),
                    distribution={}
                ),
                communication_style=CommunicationStyle(),
                personality_summary=data.get("personality_summary", "大模型分析完成。")
            )
            return profile
            
        except Exception as e:
            print(f"  [LLM Profiler Error for {name}] {e}")
            return CharacterProfile(
                name=name,
                role_in_story=role,
                total_mentions=len(character_texts),
                presence_ratio=presence_ratio,
                traits=Traits(),
                emotions=EmotionProfile(),
                communication_style=CommunicationStyle(),
                personality_summary="大模型分析失败，暂无信息。"
            )

    def _infer_role(
        self,
        name: str,
        character_texts: list[str],
        all_texts: list[str],
        all_characters: list[str] = (),
    ) -> str:
        total_texts = max(len(all_texts), 1)
        # 长文常常只有 1 段，这时"出场段数/总段数"恒为 1，角色就全成了主角。
        # 有多个角色可比时，改用"出现次数占全体比重"来分档。
        if total_texts <= 1 and all_characters:
            counts = {c: sum(t.count(c) for t in all_texts) for c in all_characters}
            total_mentions = sum(counts.values()) or 1
            share = counts.get(name, 0) / total_mentions
            for role in ("protagonist", "supporting", "minor"):
                if share > self.role_thresholds[role]:
                    return role
            return "background"
        presence = len(character_texts) / total_texts
        if presence > self.role_thresholds["protagonist"]:
            return "protagonist"
        elif presence > self.role_thresholds["supporting"]:
            return "supporting"
        elif presence > self.role_thresholds["minor"]:
            return "minor"
        else:
            return "background"

    async def profile_all_async(self, characters: list[str], text: str, relations: list[Relation]) -> Dict[str, CharacterProfile]:
        return {p.name: p for p in self.profile_all(characters, [text])}


class CharacterProfiler(BaseCharacterProfiler):
    """人物性格刻画器

    通过分析人物相关的文本，刻画其性格特质、情感特征、沟通风格和在故事中的角色。
    """

    def __init__(self):
        # 性格特质词典（基于大五人格 + 中文语境扩展）
        self.trait_lexicon: dict[str, list[str]] = {
            "leadership": ["决定", "安排", "组织", "领导", "负责", "主导", "指挥", "管理"],
            "creativity": ["想法", "创新", "建议", "提出", "新方案", "设计", "创意"],
            "agreeableness": ["同意", "支持", "帮助", "理解", "感谢", "配合", "协助"],
            "conscientiousness": ["仔细", "认真", "检查", "确保", "详细", "严谨", "核对"],
            "extraversion": ["活跃", "发言", "讨论", "分享", "表达", "交流", "沟通"],
            "emotional_stability": ["冷静", "理性", "客观", "平和", "稳定", "淡定"],
            "assertiveness": ["坚持", "要求", "必须", "一定", "强调", "主张"],
            "cooperativeness": ["合作", "配合", "一起", "共同", "团队"],
        }

        # 情感词典
        self.emotion_lexicon: dict[str, list[str]] = {
            "positive": ["满意", "高兴", "成功", "优秀", "棒", "好", "赞", "完美", "顺利"],
            "negative": ["担心", "困难", "问题", "失败", "糟糕", "差", "遗憾", "麻烦"],
            "anger": ["生气", "愤怒", "不满", "反对", "讨厌", "恼火"],
            "fear": ["担心", "害怕", "焦虑", "紧张", "恐慌", "不安"],
            "joy": ["开心", "高兴", "兴奋", "满意", "喜悦", "愉快"],
            "sadness": ["难过", "伤心", "失望", "遗憾", "悲哀"],
        }

        # 角色类型推断阈值
        self.role_thresholds = {
            "protagonist": 0.3,      # 出现频率 > 30%
            "supporting": 0.2,        # 出现频率 > 20%
            "minor": 0.1,             # 出现频率 > 10%
            "background": 0.0,        # 其余
        }

    def profile_character(
        self, name: str, texts: list[str], all_characters: Optional[list[str]] = None
    ) -> CharacterProfile:
        """刻画单个人物

        Args:
            name: 人物名称
            texts: 所有文本
            all_characters: 所有人物列表（用于角色推断）

        Returns:
            CharacterProfile 对象
        """
        # 收集该人物相关的文本
        character_texts = [t for t in texts if name in t]
        
        traits_dict = self._analyze_traits(character_texts)
        emotions_dict = self._analyze_emotions(character_texts)
        comm_style_dict = self._analyze_communication_style(character_texts)
        
        role = self._infer_role(name, character_texts, texts, all_characters or [])

        profile = CharacterProfile(
            name=name,
            role_in_story=role,
            total_mentions=len(character_texts),
            presence_ratio=len(character_texts) / max(len(texts), 1),
            traits=Traits(**traits_dict),
            emotions=EmotionProfile(**emotions_dict),
            communication_style=CommunicationStyle(**comm_style_dict)
        )

        # 性格总结依赖于 profile 对象本身
        profile.personality_summary = self._summarize_personality(profile)

        return profile

    def profile_all(
        self, characters: list[str], texts: list[str]
    ) -> List[CharacterProfile]:
        """批量刻画所有人物

        Args:
            characters: 人物列表
            texts: 所有文本

        Returns:
            CharacterProfile 对象列表
        """
        return [self.profile_character(c, texts, characters) for c in characters]

    async def profile_all_async(self, characters: list[str], text: str, relations: list[Relation]) -> Dict[str, CharacterProfile]:
        return {c: self.profile_character(c, [text], characters) for c in characters}

    def _analyze_traits(self, texts: list[str]) -> dict[str, float]:
        """分析性格特质"""
        if not texts:
            return {trait: 0.0 for trait in self.trait_lexicon}

        trait_scores: dict[str, float] = {trait: 0.0 for trait in self.trait_lexicon}

        for text in texts:
            for trait, keywords in self.trait_lexicon.items():
                for keyword in keywords:
                    if keyword in text:
                        trait_scores[trait] += 1

        # 归一化到 [0, 1]
        total = sum(trait_scores.values())
        if total > 0:
            trait_scores = {k: round(v / total, 3) for k, v in trait_scores.items()}

        return trait_scores

    def _analyze_emotions(self, texts: list[str]) -> dict[str, Any]:
        """分析情感特征"""
        if not texts:
            return {
                "distribution": {e: 0 for e in self.emotion_lexicon},
                "dominant": "neutral",
                "valence": "neutral",
            }

        emotion_counts: Counter = Counter()
        for text in texts:
            for emotion, keywords in self.emotion_lexicon.items():
                for keyword in keywords:
                    if keyword in text:
                        emotion_counts[emotion] += 1

        distribution = dict(emotion_counts)

        # 确定主导情感
        if emotion_counts:
            dominant_emotion = emotion_counts.most_common(1)[0][0]
        else:
            dominant_emotion = "neutral"

        # 情感效价
        pos_count = emotion_counts.get("positive", 0) + emotion_counts.get("joy", 0)
        neg_count = (
            emotion_counts.get("negative", 0)
            + emotion_counts.get("anger", 0)
            + emotion_counts.get("fear", 0)
            + emotion_counts.get("sadness", 0)
        )

        if pos_count > neg_count:
            valence = "positive"
        elif neg_count > pos_count:
            valence = "negative"
        else:
            valence = "neutral"

        return {
            "distribution": distribution,
            "dominant": dominant_emotion,
            "valence": valence,
        }

    def _analyze_communication_style(self, texts: list[str]) -> dict[str, Any]:
        """分析沟通风格"""
        if not texts:
            return {
                "verbosity": 0.0,
                "frequency": 0,
                "initiation": 0,
                "responsiveness": 0,
            }

        return {
            "verbosity": round(sum(len(t) for t in texts) / len(texts), 1),
            "frequency": len(texts),
            "initiation": sum(
                1 for t in texts if any(t.strip().startswith(w) for w in ["我", "建议", "认为", "觉得"])
            ),
            "responsiveness": sum(
                1 for t in texts if any(w in t for w in ["同意", "反对", "补充", "回应"])
            ),
        }

    def _infer_role(
        self,
        name: str,
        character_texts: list[str],
        all_texts: list[str],
        all_characters: list[str],
    ) -> str:
        """推断人物在故事中的角色"""
        total_texts = max(len(all_texts), 1)

        # 长文常常只有 1 段，这时"出场段数/总段数"恒为 1，所有角色都会变成主角。
        # 有多个角色可比时，改用"出现次数占全体比重"分档，段数维度只做兜底。
        if total_texts <= 1 and all_characters:
            counts = {c: sum(t.count(c) for t in all_texts) for c in all_characters}
            total_mentions = sum(counts.values()) or 1
            share = counts.get(name, 0) / total_mentions
            for role in ("protagonist", "supporting", "minor"):
                if share > self.role_thresholds[role]:
                    return role
            return "background"

        presence = len(character_texts) / total_texts

        # 检查被他人提及的频率
        mentions_by_others = 0
        for text in all_texts:
            if name in text:
                # 检查是否有其他人物也在同一文本中
                other_chars = [c for c in all_characters if c != name and c in text]
                if other_chars:
                    mentions_by_others += 1

        # 被提及比 = 被他人提及 / 本人出现
        mention_ratio = mentions_by_others / max(len(character_texts), 1)

        if presence > self.role_thresholds["protagonist"]:
            return "protagonist"
        elif presence > self.role_thresholds["supporting"]:
            if mention_ratio > 1.5:
                return "supporting"  # 高频被提及的配角
            return "supporting"
        elif presence > self.role_thresholds["minor"]:
            return "minor"
        else:
            return "background"

    def _summarize_personality(self, profile: CharacterProfile) -> str:
        """生成性格总结"""
        traits = profile.traits.model_dump()
        emotions = profile.emotions.model_dump()
        role = profile.role_in_story

        # 检查是否有非零的特质
        if not any(traits.values()):
            return "暂无足够信息"

        # 找出最突出的 3 个特质
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        top_traits = [t for t, s in sorted_traits[:3] if s > 0]

        role_names = {
            "protagonist": "主角",
            "supporting": "重要配角",
            "minor": "次要角色",
            "background": "背景角色",
        }

        parts = [f"在故事中扮演{role_names.get(role, role)}"]
        if top_traits:
            trait_name_map = {
                "leadership": "领导力强",
                "creativity": "有创造力",
                "agreeableness": "随和合作",
                "conscientiousness": "尽责认真",
                "extraversion": "外向活跃",
                "emotional_stability": "情绪稳定",
                "assertiveness": "有主见",
                "cooperativeness": "合作导向",
            }
            trait_desc = "、".join(trait_name_map.get(t, t) for t in top_traits)
            parts.append(f"性格特征：{trait_desc}")

        if emotions:
            valence = emotions.get("valence", "neutral")
            valence_map = {"positive": "积极乐观", "negative": "偏向消极", "neutral": "情感中性"}
            parts.append(f"情感倾向：{valence_map.get(valence, valence)}")

        return "；".join(parts)
