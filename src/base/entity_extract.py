"""从正文里自动抽人物：jieba 的 nr 标签 + 三个过滤信号。

jieba 的 nr 会把「巴掌/明白/冷笑/小镇/金光」也标成人名，直接拿来做词典会把
关系网络污染成"叶凡 →[association]→ 巴掌"这种。这里用三个互补信号过滤：

1. **说话归属**：中文小说里人物几乎一定会「说/道/问/笑/喝/叹」，
   统计"候选 + 引语动词"的次数，是最强的信号。
2. **姓氏**：2~4 字且首字是常见姓氏（复用 alias 模块的姓氏表）。
3. **动作主体**：候选后面紧跟「看着/望向/走进/转身/点头」这类动作的概率。

同时保留一个按作品的可配置白名单/黑名单，方便人工纠正。
"""
from __future__ import annotations

import re
from collections import Counter

import jieba.posseg as pseg

from .alias import CharacterAliasResolver
from .entity_filter import filter_persons

# 引语动词（说话归属）
SPEECH_VERBS = (
    "说道", "说", "道", "问道", "问", "答道", "答", "笑道", "笑", "喝道", "喝",
    "叹道", "叹", "叫道", "叫", "嚷", "低声道", "沉声道", "淡淡道", "开口道",
)
# 动作主体（人做主语）
ACTION_VERBS = (
    "看着", "望向", "走进", "转身", "点头", "摇头", "皱眉", "伸手", "站起",
    "坐下", "抬头", "低头", "回头", "开口", "上前", "后退", "冲", "拉住",
)

_SPEECH_RE = re.compile("(%s)" % "|".join(sorted(SPEECH_VERBS, key=len, reverse=True)))
_ACTION_RE = re.compile("(%s)" % "|".join(sorted(ACTION_VERBS, key=len, reverse=True)))


def _surnames() -> set[str]:
    # alias.py 里有一张常见姓氏表，直接复用，避免两处维护
    return CharacterAliasResolver().common_surnames


def collect_stats(text: str, max_chars: int = 1_200_000) -> dict[str, dict]:
    """扫一遍文本，统计每个 nr 候选的三个信号。

    额外记录"同一串在别处被标成非人名"的次数——jieba 会把「冷笑」「明白」
    「巴掌」这类词在某些语境标成 nr，但同一个串在别处会标成 v/n，
    这种多标签词基本可以判死。
    """
    text = text[:max_chars]
    mentions: Counter[str] = Counter()
    non_person_hits: Counter[str] = Counter()

    for word, flag in pseg.cut(text):
        w = word.strip()
        if flag == "nr":
            name = w
            if not (2 <= len(name) <= 4) or re.search(r"[0-9０-９]", name):
                continue
            mentions[name] += 1
        elif flag and flag[0] in "vna":  # 动词/名词/形容词/副词
            if 2 <= len(w) <= 4:
                non_person_hits[w] += 1

    speech: Counter[str] = Counter()
    action: Counter[str] = Counter()
    for name in mentions:
        tail = re.escape(name)
        speech[name] = len(re.findall(tail + r"[^。！？\n]{0,2}?(" + _SPEECH_RE.pattern + r")", text))
        action[name] = len(re.findall(tail + r"[^。！？\n]{0,2}?(" + _ACTION_RE.pattern + r")", text))

    return {
        n: {
            "mentions": mentions[n],
            "speech": speech[n],
            "action": action[n],
            "non_person": non_person_hits[n],
        }
        for n in mentions
    }


# 明显的非人物后缀（家族/门派/功法/器物）
NON_PERSON_SUFFIX = ("家", "门", "宗", "族", "派", "经", "诀", "术", "法", "光", "声", "音", "经书")


def _is_substring_of_longer(name: str, all_names: set[str]) -> bool:
    """「姬紫」是「姬紫月」的碎片时该合并到长名，而不是当独立人物。"""
    return any(name != other and name in other for other in all_names)


def pick_characters(
    stats: dict[str, dict],
    book: str = "",
    min_mentions: int = 30,
    min_speech: int = 3,
    require_surname: bool = False,
    extra_ban: set[str] | None = None,
    extra_keep: set[str] | None = None,
) -> list[str]:
    """按信号挑人。

    Args:
        stats: collect_stats 的输出
        book: 作品名，用于非人物停用表
        min_mentions: 最少出现次数
        min_speech: 至少被当作一次说话人（或说话人语境）
        require_surname: 是否强制首字为常见姓氏（网文里有"黑皇""段德"这类，默认关）
        extra_ban / extra_keep: 人工纠正

    Returns:
        按出现次数降序的人物列表
    """
    surnames = _surnames()
    keep = set(extra_keep or ())
    picked: list[str] = []
    for name, st in stats.items():
        if name in keep:
            picked.append(name)
            continue
        if st["mentions"] < min_mentions:
            continue
        if st["speech"] < min_speech and st["action"] < min_speech:
            continue
        if require_surname and name[0] not in surnames:
            continue
        # 多标签词（同一个串在别处被标成动词/名词）基本不是人名
        if st["non_person"] >= max(3, st["mentions"] * 0.1):
            continue
        if name.endswith(NON_PERSON_SUFFIX):
            continue
        picked.append(name)

    # 碎片名并入长名：只保留"不是别人子串"的那些
    all_picked = set(picked)
    picked = [n for n in picked if not _is_substring_of_longer(n, all_picked)]

    picked = filter_persons(picked, book, extra_ban)
    picked.sort(key=lambda n: -stats[n]["mentions"])
    return picked
