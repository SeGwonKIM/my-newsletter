"""소스 후보 측정 — audience.yaml 의 후보 전부를 관문 네 개로 잰다.

  G1 접근    : HTTP 200 이고 피드 항목이 있다
  G2 최신성  : 최근 24시간 안에 올라온 글이 있다
  G3 본문    : 최근 글 3건 중 3건 모두 원문 본문 600자 이상 추출된다
  G4 적합도  : 최근 글 제목 중 독자 키워드가 들어간 비율이 25% 이상이다

결과는 화면과 store/sources.md 에 남긴다. 실행: python tools/measure_sources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feedparser  # noqa: E402
import requests  # noqa: E402
from graph import CONFIG, UA, entry_time, extract_body, is_relevant, now_utc  # noqa: E402

MIN_BODY, FIT_MIN = 600, 0.25


def measure(name: str, url: str) -> dict:
    row = {"name": name, "status": "-", "entries": 0, "new24": 0, "body": "-", "fit": 0.0}
    try:
        r = requests.get(url, headers=UA, timeout=20)
    except requests.RequestException as e:
        row["status"] = type(e).__name__
        return row
    row["status"] = r.status_code
    entries = feedparser.parse(r.content).entries
    row["entries"] = len(entries)
    if not entries:
        return row
    now = now_utc()
    times = [t for t in (entry_time(e, now)[0] for e in entries) if t]
    row["new24"] = sum((now - t).total_seconds() <= 24 * 3600 for t in times)
    ok = sum(bool((b := extract_body(e.link)) and len(b) >= MIN_BODY) for e in entries[:3])
    row["body"] = f"{ok}/{min(3, len(entries))}"
    titles = [e.get("title", "") for e in entries[:30]]
    row["fit"] = sum(is_relevant(t) for t in titles) / len(titles)
    return row


def verdict(row: dict) -> tuple[str, str]:
    if row["status"] != 200 or row["entries"] == 0:
        return "탈락", "G1 접근 실패"
    if row["new24"] == 0:
        return "탈락", "G2 최근 24시간 글 없음"
    if row["body"] != f"{min(3, row['entries'])}/{min(3, row['entries'])}":
        return "탈락", f"G3 본문 추출 {row['body']}"
    if row["fit"] < FIT_MIN:
        return "탈락", f"G4 적합도 {row['fit']:.0%}"
    return "채택", "관문 4개 통과"


def main() -> None:
    lines = ["| 소스 | HTTP | 항목 | 24h 신규 | 본문추출 | 적합도 | 판정 | 근거 |",
             "|---|---|---|---|---|---|---|---|"]
    for s in CONFIG["sources"]:
        row = measure(s["name"], s["url"])
        v, why = verdict(row)
        line = (f"| {row['name']} | {row['status']} | {row['entries']} | {row['new24']} | "
                f"{row['body']} | {row['fit']:.0%} | {v} | {why} |")
        print(line)
        lines.append(line)
    out = Path(__file__).resolve().parents[1] / "store" / "sources.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
