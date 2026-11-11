"""别名合并的可配置规则。

原来的别名识别只看"姓氏 + 称谓 / 小X / 老X / 阿X"这类构词法，
覆盖不到 `诺诺 → 陈墨瞳`（昵称完全不同）、`路鸣泽 → 路明非`（同一角色的另一面）
这类关系，也覆盖不到"地图里的两个候选指同一个人"。

这里提供三件东西：
1. `build_explicit_mapping()`：从角色映射表生成 pipeline 要的 explicit_mapping，
   并校验"一个别名只能指向一个人"，冲突时报错而不是静默取第一个。
2. `merge_into_canonical()`：把角色列表里的别名并到规范名之下。
3. `detect_alias_conflicts()`：给出冲突清单，供人工裁决（而不是自动猜）。
"""
from __future__ import annotations

from collections import defaultdict


def build_explicit_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    """把 {规范名: [别名...]} 规整成 pipeline 用的形式。

    - 别名去重、去掉与规范名相同的项、去掉空串
    - 同一别名指向多个规范名时抛 ValueError（说明映射表写错了）
    """
    result: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        if not canonical:
            continue
        clean: list[str] = []
        for alias in aliases or []:
            alias = (alias or "").strip()
            if not alias or alias == canonical:
                continue
            prev = owner.get(alias)
            if prev and prev != canonical:
                raise ValueError(
                    f"别名冲突：{alias!r} 同时指向 {prev!r} 和 {canonical!r}，请先修订映射表"
                )
            owner[alias] = canonical
            clean.append(alias)
        if canonical in owner:
            raise ValueError(f"规范名 {canonical!r} 同时被当作 {owner[canonical]!r} 的别名")
        result[canonical] = clean
    return result


def merge_into_canonical(characters: list[str], mapping: dict[str, list[str]]) -> list[str]:
    """把角色列表里的别名并到规范名，返回去重后的规范名列表。

    规范名本身没出现、只有别名出现时，也用规范名占位（保持顺序按首次出现）。
    """
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        for alias in aliases:
            alias_to_canonical[alias] = canonical

    seen: set[str] = set()
    merged: list[str] = []
    for name in characters:
        canonical = alias_to_canonical.get(name, name)
        if canonical in seen:
            continue
        seen.add(canonical)
        merged.append(canonical)
    return merged


def detect_alias_conflicts(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    """返回 {别名: [指向它的规范名...]}，只列出有冲突的条目。"""
    owners: dict[str, list[str]] = defaultdict(list)
    for canonical, aliases in mapping.items():
        for alias in aliases or []:
            owners[alias].append(canonical)
    return {alias: names for alias, names in owners.items() if len(names) > 1}
