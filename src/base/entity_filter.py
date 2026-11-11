"""实体类型过滤：把人名词典里的地名/组织名/专有物名挡在"人物"之外。

规则 NER 只做子串匹配，词典里只要混进 `卡塞尔`（学院名）就会被当成人物节点，
进而参与关系网络与画像。这里给一份可按作品覆盖的停用表，默认覆盖常见几部。
"""
from __future__ import annotations

# 通用停用：这些词出现在人名列表里几乎一定是误判
GENERIC_NON_PERSON = {
    "学校", "学院", "大学", "公司", "集团", "政府", "医院", "银行",
    "天下", "世界", "人间", "江湖", "江湖人",
}

# 按作品：明确不是人物的专有名词
BOOK_NON_PERSON = {
    "longzu": {
        "卡塞尔", "卡塞尔学院", "芝加哥", "仕兰", "仕兰中学", "三峡", "日本",
        "白王", "黑王", "青铜与火之王", "龙王", "言灵", "狮心会", "学生会",
        "东京", "北京", "美国",
    },
    "hongloumeng": {
        "大观园", "荣国府", "宁国府", "贾府", "潇湘馆", "蘅芜苑", "怡红院",
        "稻香村", "荣府", "宁府", "贾家",
    },
}


def non_person_names(book: str = "", extra=None) -> set[str]:
    """返回该作品下不应被当作人物的名称集合。"""
    names = set(GENERIC_NON_PERSON)
    names |= BOOK_NON_PERSON.get((book or "").lower(), set())
    if extra:
        names |= set(extra)
    return names


def filter_persons(names, book: str = "", extra=None) -> list[str]:
    """从候选名单里剔除非人物，保持原有顺序。"""
    banned = non_person_names(book, extra)
    return [n for n in names if n and n not in banned]


def is_person(name: str, book: str = "", extra=None) -> bool:
    """单个名称判定。"""
    return name in filter_persons([name], book, extra)
