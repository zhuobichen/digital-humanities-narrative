"""抓《龙族1·火之晨曦》前若干章正文（公开在线阅读页），供本地流水线测试。

- 只取前 N 章（默认 6 章）作为样本，不下载全书、不长期保存。
- 版权归原作者与出版方；测试完可直接删除 data/龙族 目录。
"""
import re
import sys
from pathlib import Path
import urllib.request

OUT = Path(__file__).parent / "data" / "龙族"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "https://www.95590.org/xuanhuan/202208/14656{}.html"  # 146560 -> 146567
MAX_CHAPTERS = 6


def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # 单章失败不中断
        print(f"  [x] {url}: {e}", file=sys.stderr)
        return None


def extract_text(html: str) -> str:
    m = re.search(r'class="entry-content">(.*?)</article>', html, re.S)
    body = m.group(1) if m else html
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<br\s*/?>|</p>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = body.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    body = body.replace("&lt;", "<").replace("&gt;", ">")
    lines = [ln.strip() for ln in body.split("\n")]
    return "\n".join(ln for ln in lines if ln and not re.search(r"本章未完|点击下一页|99990|请记住本站", ln))


def main():
    chunks = []
    for i in range(MAX_CHAPTERS):
        url = BASE.format(i)
        html = fetch(url)
        if not html:
            continue
        title = re.search(r"<title>(.*?)</title>", html)
        text = extract_text(html)
        print(f"[{i}] {(title.group(1) if title else url)[:40]}  {len(text)} 字")
        if len(text) > 200:
            chunks.append(text)
    if not chunks:
        print("[x] 没抓到正文", file=sys.stderr)
        return 1
    out = OUT / "龙族1_前6章.txt"
    out.write_text("\n\n".join(chunks), "utf-8")
    total = sum(len(c) for c in chunks)
    print(f"\n[OK] {out}  {len(chunks)} 章 / {total} 字")
    return 0


if __name__ == "__main__":
    sys.exit(main())
