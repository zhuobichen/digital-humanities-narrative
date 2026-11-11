"""无 spaCy 时的词典后端，接口与 NamedEntityRecognizer 对齐。

作用：让 NER 在没装 spaCy 中文模型的机器上也能工作（纯规则、零下载），
同时让 test_ner 这类直接构造 `backend="spacy"` 的调用不至于 ImportError。
"""
from __future__ import annotations


class DictBackendRecognizer:
    """只依赖人名词典的实体识别后端。"""

    def __init__(self, character_names=None, strict_dict_mode: bool = True):
        self.character_names = set(character_names or [])
        self.strict_dict_mode = strict_dict_mode
        self.backend = "dict"
        self._nlp = None

    def add_character_names(self, names):
        self.character_names.update(names)

    def extract_characters(self, text: str):
        """按词典（长名优先）扫描文本中出现的人物。"""
        results = set()
        for name in sorted(self.character_names, key=len, reverse=True):
            if name and name in text:
                results.add(name)
        return sorted(results)
