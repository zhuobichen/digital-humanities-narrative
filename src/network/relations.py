"""人物关系抽取器

从文本中识别并分类人物之间的关系。
"""

import json
import hashlib
from typing import Any, List
from pydantic import BaseModel
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import Relation
from ..interfaces import BaseRelationshipExtractor
from .sentence_relations import extract_sentence_relations

class LLMRelationshipExtractor(BaseRelationshipExtractor):
    """基于大模型的人物关系抽取器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # 初始化本地磁盘缓存，用于断点续传
        self.cache = Cache(".llm_cache/relations")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(Exception)
    )
    def _call_llm_with_retry(self, prompt: str) -> str:
        """带重试机制和断点续传缓存的 LLM 调用"""
        # 使用 Prompt 和模型参数的组合哈希作为缓存 Key，防止未来更换模型时读取了旧缓存
        cache_key_content = f"deepseek-chat_temp0.1_{prompt}"
        prompt_hash = hashlib.md5(cache_key_content.encode('utf-8')).hexdigest()
        
        if prompt_hash in self.cache:
            return self.cache[prompt_hash]
            
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=30 # 增加超时保护
        )
        content = response.choices[0].message.content
        
        # 预验证 JSON 格式，如果 LLM 返回了损坏的 JSON，则抛出异常触发重试，并且不存入缓存
        try:
            json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM 返回了无效的 JSON 格式: {content[:100]}...")
            
        self.cache[prompt_hash] = content # 存入缓存
        return content

    def extract_relations_with_sentiment(self, text: str, characters: list[str]) -> List[Relation]:
        """使用大模型抽取关系，附带情感倾向"""
        # 优化：采用滑动窗口分块，防止边界关系被截断
        chunk_size = 4000
        overlap = 400
        relations = []
        seen_pairs = set()
        
        # 构造滑动窗口
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((start, text[start:end]))
            start += chunk_size - overlap
            
        from rich.progress import Progress
        
        print(f"  [LLM] 文本已被切分为 {len(chunks)} 块(带重叠)，开始深度抽取 (支持断点续传)...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        lock = threading.Lock()
        
        def process_chunk(i, chunk):
            # 过滤出当前 chunk 实际出现的人物，减少 LLM 负担
            chars_in_chunk = [c for c in characters if c in chunk]
            if len(chars_in_chunk) < 2:
                return []
                
            prompt = f"""
你是一个专业的古典文学数字人文分析专家。请从以下文本片段中提取人物之间的关系。
已知在这个片段中出现的人物有: {chars_in_chunk}

请提取他们之间的实质性互动关系，并输出为 JSON 数组。
关系类型(type)必须从以下选项中选择一个:
- hierarchical: 层级关系 (如上下级、长辈晚辈)
- collaborative: 协作关系 (如合作、一起做事)
- conflict: 冲突关系 (如争吵、对抗)
- social: 社交关系 (如聊天、聚会)
- association: 其他普通关联

每个关系必须包含以下字段:
- source: 人物1名称
- target: 人物2名称
- type: 关系类型
- context: 证明该关系的简短原文上下文(20字以内)
- sentiment: 情感倾向，必须是 'positive'(亲密/友好)、'negative'(敌对/冲突) 或 'neutral'(中立) 之一。
- location: 该互动发生的地点或场景名称（如：卡塞尔学院、诺顿馆、网吧、冰海）。如果文本中未提及具体地点，请填写"未知"。

请直接输出 JSON 数组格式（不要包裹在 Markdown 代码块中），如果没有提取到任何关系，请输出 []。

文本片段:
{chunk}
"""
            chunk_relations = []
            try:
                content = self._call_llm_with_retry(prompt)
                
                # json_object 模式下 deepseek 可能会返回一个包含 array 的 dict，或者直接 array
                # 为了兼容，尝试解析
                data = json.loads(content)
                
                # 处理返回格式的差异
                if isinstance(data, dict):
                    # 寻找第一个列表类型的值
                    for v in data.values():
                        if isinstance(v, list):
                            data = v
                            break
                    if isinstance(data, dict):
                        data = []
                        
                # --- 多智能体自我纠错 (Agentic Workflow - Reviewer Agent) ---
                if data and len(data) > 0:
                    review_prompt = f"""
你是一个极其严格的数字人文事实审查员 (Reviewer Agent)。
你的任务是审查由提取器(Extractor)刚刚从以下原文片段中提取出的人物关系。

【原文片段】：
{chunk}

【待审查的关系数组 (JSON)】：
{json.dumps(data, ensure_ascii=False)}

请严格执行以下审查：
1. 幻觉剔除：如果在原文片段中找不到明确的依据，必须果断删除该关系。
2. 极性修正：检查 sentiment 是否准确反映了片段中的情感，且值必须为 'positive', 'negative' 或 'neutral'。
3. 证据核实：确保 context 字段准确引用了片段中的原文。
4. 地点核实：确保 location 字段准确反映了原文中的地点，不可凭空捏造。如果互动并未发生在该提取地点，必须删除或修正该关系。

请直接输出修正后的 JSON 数组（不要包裹在 Markdown 代码块中），格式与待审查结果完全一致。如果所有提取都是错误的，请输出 []。
"""
                    try:
                        review_content = self._call_llm_with_retry(review_prompt)
                        corrected_data = json.loads(review_content)
                        if isinstance(corrected_data, dict):
                            for v in corrected_data.values():
                                if isinstance(v, list):
                                    corrected_data = v
                                    break
                            if isinstance(corrected_data, dict):
                                corrected_data = []
                        data = corrected_data
                    except Exception as e:
                        print(f"  [Reviewer Agent Error at chunk {i}] {e}，降级使用初始提取结果")
                # -----------------------------------------------------------------
                
                for item in data:
                    src = item.get("source")
                    tgt = item.get("target")
                    if src in characters and tgt in characters and src != tgt:
                        pair = tuple(sorted([src, tgt]))
                        
                        # 改为在列表中合并同一对人物的关系（如果类型相同则认为是同一次互动被重复提取）
                        # 这里我们不仅用 pair 区分，还要结合上下文粗略排重，或者允许同一对人物存在多次不同位置的互动
                        # 从而在后续计算动态关系时有更多时间序列数据点
                        # 为了避免滑动窗口重叠区完全相同的提取，我们利用上下文(context)的简单哈希来去重
                        context_snippet = item.get("context", "")
                        unique_interaction_key = f"{pair[0]}_{pair[1]}_{context_snippet[:10]}"
                        
                        with lock:
                            if unique_interaction_key not in seen_pairs:
                                seen_pairs.add(unique_interaction_key)
                                chunk_relations.append(Relation(
                                    source=src,
                                    target=tgt,
                                    type=item.get("type", "association"),
                                    context=context_snippet[:20],
                                    position=i,  # 粗略位置
                                    sentiment=item.get("sentiment", "neutral"),
                                    location=item.get("location", "未知")
                                ))
            except Exception as e:
                print(f"  [LLM Extract Error at chunk {i}] {e}")
                
            return chunk_relations

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in chunks]
            for future in as_completed(futures):
                relations.extend(future.result())
                
        return relations

class RelationshipExtractor(BaseRelationshipExtractor):
    """人物关系抽取器

    通过关键词模式匹配和共现分析，从文本中抽取人物关系。
    支持层级、协作、冲突、社交四种关系类型。
    """

    def __init__(self):
        # 关系类型定义
        self.relation_patterns: dict[str, dict[str, Any]] = {
            "hierarchical": {
                "patterns": [
                    "是.*的下属", "向.*汇报", "管理", "领导",
                    "负责", "安排", "任命", "上级", "下属",
                ],
                "keywords": ["经理", "主管", "下属", "汇报", "领导", "上级", "安排"],
                "description": "层级关系",
            },
            "collaborative": {
                "patterns": [
                    "和.*一起", "与.*合作", "和.*讨论", "共同",
                    "配合", "协助", "协作",
                ],
                "keywords": ["合作", "讨论", "开会", "一起", "配合", "协同"],
                "description": "协作关系",
            },
            "conflict": {
                "patterns": [
                    "和.*争论", "与.*冲突", "反对", "不同意",
                    "指责", "批评", "质疑", "矛盾",
                ],
                "keywords": ["争论", "冲突", "反对", "不同意", "批评", "指责"],
                "description": "冲突关系",
            },
            "social": {
                "patterns": [
                    "和.*吃饭", "与.*聚会", "和.*聊天", "一起.*玩",
                    "约", "见面", "拜访",
                ],
                "keywords": ["吃饭", "聚会", "聊天", "朋友", "见面", "约"],
                "description": "社交关系",
            },
        }

    def extract_relations(
        self, text: str, characters: list[str], min_cooccur: int = 1, directed: bool = True
    ) -> List[Relation]:
        """从文本中抽取人物关系

        Args:
            text: 输入文本
            characters: 人物列表（已合并别名的规范名）
            min_cooccur: 同一关系至少出现几次才保留，默认 1
            directed: True 用"主体—动词—客体"语序定方向（`directed_relations`），
                False 退回句内共现的句级版本（`sentence_relations`）

        Returns:
            Relation 对象列表，按权重降序
        """
        if directed:
            from .directed_relations import extract_directed_relations

            extracted = extract_directed_relations(text, characters, min_cooccur)
        else:
            extracted = extract_sentence_relations(text, characters, min_cooccur)
        return [
            Relation(
                source=item["source"],
                target=item["target"],
                type=item["type"],
                context_snippet=item["evidence"],
                context=item["evidence"],
                position=idx,
                location="未知",
            )
            for idx, item in enumerate(extracted)
        ]

    def _extract_relations_by_window(
        self, text: str, characters: list[str]
    ) -> List[Relation]:
        """旧的 100 字滑动窗口共现（保留作对比基线，默认不再调用）。"""
        relations: List[Relation] = []
        window_size = 100  # 字符窗口

        # 滑动窗口检测共现
        step = 20
        for i in range(0, max(len(text) - window_size, 1), step):
            window = text[i : i + window_size]

            # 检测窗口中的人物
            chars_in_window = [c for c in characters if c in window]
            if len(chars_in_window) < 2:
                continue

            # 检测关系类型
            relation_type = self._detect_relation_type(window)

            # 为窗口中的每对人物创建关系
            for j in range(len(chars_in_window)):
                for k in range(j + 1, len(chars_in_window)):
                    relations.append(Relation(
                        source=chars_in_window[j],
                        target=chars_in_window[k],
                        type=relation_type,
                        # context_snippet 是"能指回原文的那句话"。原来只有 LLM 分支
                        # 赋值，规则分支恒为空串，下游证据栏/RAG 提示永远是空。
                        context_snippet=window.strip()[:120],
                        context=window.strip()[:200],
                        position=i,
                        location="未知"
                    ))

        # 去重：同一对人物保留证据最强的 N 条，而不是只留首次出现——
        # 只留 1 条会把后续互动（态度反转、冲突）全丢掉。
        max_per_pair = 2
        kept: dict[tuple[str, str], int] = {}
        unique_relations = []
        for rel in relations:
            pair = tuple(sorted([rel.source, rel.target]))
            n = kept.get(pair, 0)
            if n >= max_per_pair:
                continue
            kept[pair] = n + 1
            unique_relations.append(rel)

        return unique_relations

    def _detect_relation_type(self, text: str) -> str:
        """检测窗口内的主要关系类型

        按关键词匹配数最多确定类型。
        """
        scores: dict[str, int] = {}
        for rel_type, config in self.relation_patterns.items():
            keywords = config.get("keywords", [])
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[rel_type] = score

        if scores:
            return max(scores, key=scores.get)
        return "association"  # 默认：普通关联

    def extract_relations_with_sentiment(
        self, text: str, characters: list[str]
    ) -> List[Relation]:
        """抽取关系并附带情感分析

        Args:
            text: 输入文本
            characters: 人物列表

        Returns:
            包含 sentiment 属性的 Relation 列表
        """
        relations = self.extract_relations(text, characters)

        # 简单情感分析
        positive_words = {"满意", "高兴", "成功", "好", "棒", "赞", "感谢", "支持"}
        negative_words = {"不满", "生气", "失败", "差", "糟", "问题", "担心", "反对"}

        for rel in relations:
            context = rel.context
            pos_count = sum(1 for w in positive_words if w in context)
            neg_count = sum(1 for w in negative_words if w in context)
            total = pos_count + neg_count
            if total == 0:
                rel.sentiment = "neutral"
            else:
                sentiment = (pos_count - neg_count) / total  # [-1, 1]
                rel.sentiment = (
                    "positive" if sentiment > 0 else ("negative" if sentiment < 0 else "neutral")
                )

        return relations

    def describe_relation(self, relation_type: str) -> str:
        """获取关系类型的中文描述"""
        for rt, config in self.relation_patterns.items():
            if rt == relation_type:
                return config.get("description", relation_type)
        return "关联关系"
