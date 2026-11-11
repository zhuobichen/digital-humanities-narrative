"""命名实体识别与共指消解

支持 spaCy 模型 + 自定义人名词典双重匹配，提升中文文学文本的人物识别准确率。
"""

import re
from typing import Optional


class NamedEntityRecognizer:
    """中文命名实体识别器

    支持 spaCy 和 Hugging Face 两种后端，同时支持自定义人名词典增强。
    """

    def __init__(
        self,
        backend: str = "spacy",
        model_name: Optional[str] = None,
        character_names: Optional[list[str]] = None,
        strict_dict_mode: bool = False,
    ):
        """
        Args:
            backend: "spacy" 或 "transformers"
            model_name: 模型名称，默认使用中文模型
            character_names: 自定义人物名称白名单
            strict_dict_mode: 如果为 True，只使用字典提取人物，跳过模型 NER
        """
        self.backend = backend
        self._nlp = None
        self.character_names = set(character_names or [])
        self.strict_dict_mode = strict_dict_mode

        if strict_dict_mode:
            return  # Skip loading models

        if backend == "spacy":
            # spaCy 与中文模型是可选依赖：缺了不该让整条流水线起不来，
            # 降级成纯词典后端（同时把 backend 改成 dict，免得后续分支用错）。
            try:
                import spacy

                self._nlp = spacy.load(model_name or "zh_core_web_sm")
                self._nlp.max_length = 2000000  # 增加 max_length 以支持长文本
            except (ImportError, OSError) as exc:
                import warnings

                warnings.warn(
                    f"spaCy 后端不可用（{exc.__class__.__name__}: {exc}），降级为词典模式。"
                    "启用模型：pip install spacy && python -m spacy download zh_core_web_sm",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.backend = "dict"
                self._nlp = None
        elif backend == "transformers":
            self.model_name = model_name or "ckiplab/bert-base-chinese-ner"     
        else:
            raise ValueError(f"不支持的后端: {backend}")

    def add_character_names(self, names: list[str]):
        """添加自定义人物名称"""
        self.character_names.update(names)

    def extract_characters(self, text: str) -> list[str]:
        """从文本中提取人物实体

        先用人名词典扫描，再用 spaCy NER 补充，最后过滤误识别。
        Args:
            text: 输入文本

        Returns:
            人物名称列表
        """
        results: set[str] = set()

        # 1. 词典匹配（优先级最高）
        if self.character_names:
            for name in sorted(self.character_names, key=len, reverse=True):    
                if name in text:
                    results.add(name)

        if self.strict_dict_mode or self.backend == "dict":
            return list(results)

        # 2. spaCy NER
        ner_results = self._extract_spacy(text) if self.backend == "spacy" else self._extract_transformers(text)

        # 3. 合并 + 过滤
        for name in ner_results:
            # 过滤单个字符（通常是误识别）
            if len(name) < 2:
                continue
            # 过滤明显不是人名的词
            if self._is_false_person(name):
                continue
            results.add(name)

        # 4. 如果词典中有人名被 NER 拆分了，尝试合并
        return list(results)

    def _is_false_person(self, name: str) -> bool:
        """判断是否为误识别的人物实体"""
        # 常见误识别模式
        false_patterns = [
            r"[说道去来是的有在]" + r"$",              # "XX道" "XX说"
            r"^[这一那什么]" + r".*" + r"[时后前]$",  # 时间相关
            r"过去$", r"将来$", r"从前$", r"如今$",
            r"笑道$", r"道$", r"说道$", r"骂道$",
            r".*[来去]$",                            # 动词结尾
        ]
        for pattern in false_patterns:
            if re.search(pattern, name):
                return True
        return False

    def extract_characters_hybrid(
        self, text: str, min_name_len: int = 2
    ) -> list[str]:
        """混合模式：词典 + NER + 同姓提示

        Args:
            text: 输入文本
            min_name_len: 最少字符数

        Returns:
            人物名称列表
        """
        characters = self.extract_characters(text)
        return [c for c in characters if len(c) >= min_name_len]

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """提取所有命名实体"""
        if self.backend == "spacy":
            return self._extract_all_spacy(text)
        else:
            return {"PERSON": self._extract_transformers(text)}

    def _extract_spacy(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))

    def _extract_all_spacy(self, text: str) -> dict[str, list[str]]:
        doc = self._nlp(text)
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities

    def _extract_transformers(self, text: str) -> list[str]:
        try:
            from transformers import pipeline
            ner_pipe = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
            results = ner_pipe(text)
            return list(set(r["word"] for r in results if r.get("entity_group") == "PER"))
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers torch")

    def extract_with_coreference(self, text: str) -> list[list[str]]:
        """提取人物并做共指消解"""
        characters = self.extract_characters(text)
        clusters = self._rule_based_coref(text, characters)
        return clusters

    def _rule_based_coref(self, text: str, entities: list[str]) -> list[list[str]]:
        """基于规则的共指消解"""
        from .alias import CharacterAliasResolver
        resolver = CharacterAliasResolver()
        alias_groups = resolver.resolve_aliases(text, entities)

        resolved = set()
        clusters = []
        for group in alias_groups:
            canonical = group["canonical_name"]
            aliases = group["aliases"]
            resolved.add(canonical)
            resolved.update(aliases)
            clusters.append([canonical] + list(aliases))

        for entity in entities:
            if entity not in resolved:
                clusters.append([entity])

        return clusters
