"""나만의 뉴스레터 에이전트 — LangGraph 워크플로 본체.

  ① collect  수집   RSS 6곳에서 time window 안의 글을 모은다 (한 곳이 죽어도 계속)
  ② select   선별   예선(묶음별) → 본선(한 화면) 2단계 상대평가, 같은 사건은 하나만
  ③ report   취재   기사마다 워커를 펼쳐(Send) 본문을 읽고 요약·인사이트를 쓴다
  ④ verify   검수   숫자 대조 + LLM 판정. 불합격이면 지적을 붙여 한 번 다시 쓰고, 또 떨어지면 뺀다
  ⑤ publish  발행   out/ 에 Markdown·HTML 브리핑을 쓰고, 다시 읽어 제대로 써졌는지 확인한다

독자·기준·소스는 audience.yaml 에 있다. 실행 기록은 store/metrics.jsonl 에 한 줄씩 쌓인다.
OPENAI_API_KEY 가 없거나 --offline 이면 LLM 대신 규칙으로 돌아간다(배선 점검용, 결과에 표시).
"""

from __future__ import annotations

import html
import json
import operator
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import feedparser
import requests
import trafilatura
import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "audience.yaml").read_text(encoding="utf-8"))
NL, AUD = CONFIG["newsletter"], CONFIG["audience"]
UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}
KST = timezone(timedelta(hours=9))
MIN_BODY = 600          # 이보다 짧으면 요약할 근거가 부족하다 (1028 섹션 3 G1 기준선)


# ── 공용 도구 ─────────────────────────────────────────────────
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def entry_time(e, now: datetime) -> tuple[datetime | None, bool]:
    """게시 시각과 '보정했는가'. 국내 피드 일부는 한국 시각에 +0000 을 붙여 미래 시각이 된다."""
    t = e.get("published_parsed") or e.get("updated_parsed")
    if not t:
        return None, False
    at = datetime(*t[:6], tzinfo=timezone.utc)
    if at > now + timedelta(minutes=10):
        return at - timedelta(hours=9), True
    return at, False


def extract_body(url: str) -> str | None:
    d = trafilatura.fetch_url(url)
    return trafilatura.extract(d) if d else None


def is_relevant(title: str) -> bool:
    t = title.lower()
    return any(k.lower() in t for k in AUD["keywords"])


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def norm_url(u: str) -> str:
    return u.split("?")[0].split("#")[0].rstrip("/")


def clean_url(u: str) -> str:
    """발행용 링크 — utm_ 추적 꼬리표만 떼어 낸다 (?p=123 같은 진짜 주소는 남긴다)."""
    return u.split("?")[0] if "utm_" in u else u


# ── State ────────────────────────────────────────────────────
class Brief(TypedDict):
    hours: int
    mode: str                                   # "llm" | "offline"
    collected: list
    picked: list
    drafted: Annotated[list, operator.add]      # 취재 워커들이 나눠 채운다
    verified: list
    stats: Annotated[list, operator.add]        # metrics.jsonl 에 남길 단계별 기록
    log: Annotated[list, operator.add]


# ── LLM 층 — 온라인(OpenAI)과 오프라인(규칙)이 같은 모양을 낸다 ──
class Pick(BaseModel):
    index: int = Field(description="후보 목록에서의 번호")
    reason: str = Field(description="왜 골랐는지 독자 관점 한 문장")
    criterion: int = Field(description="해당하는 '고를 것' 기준 번호 (1부터)")
    event: str = Field(description="이 기사가 다루는 사건을 짧은 라벨로. 같은 사건이면 같은 라벨")


class Shortlist(BaseModel):
    picks: list[Pick]


class Draft(BaseModel):
    headline: str = Field(description="25자 내외 한국어 헤드라인")
    summary: str = Field(description="한국어 **딱 세 문장** 요약. 모든 문장을 ~합니다체로, 원문에 있는 사실만")
    insight: str = Field(description="1인 창업자가 이번 주에 해 볼 수 있는 구체적 행동 한두 문장. 원문 사실에 근거")
    topic: Literal["플랫폼 정책", "마케팅 전술", "지원·규제", "소비자 트렌드", "도구"]


class Verdict(BaseModel):
    ok: bool = Field(description="헤드라인과 요약이 원문에서 뒷받침되면 true")
    problems: list[str] = Field(description="원문에 없거나 틀린 부분. 없으면 빈 목록")


CRITERIA = (
    f"독자: {AUD['persona'].strip()}\n중요도 판단 문장: {AUD['importance'].strip()}\n"
    "고를 것:\n" + "\n".join(f"{i}. {k}" for i, k in enumerate(AUD["keep"], 1))
    + "\n버릴 것:\n" + "\n".join(f"- {k}" for k in AUD["drop"])
)
SYS_DRAFT = (
    f"당신은 다음 독자를 위한 뉴스레터 기자입니다.\n{AUD['persona'].strip()}\n"
    "기사 본문을 읽고 헤드라인·요약·인사이트를 **반드시 한국어로** 쓰세요. 원문이 영어여도 한국어입니다.\n"
    "요약에는 원문에 있는 사실만 쓰고, 숫자는 원문 그대로 옮기세요.\n"
    "인사이트는 '주목된다' 같은 표현 없이, 독자가 이번 주에 직접 해 볼 행동으로 쓰세요.\n"
    "한국에서 아직 쓸 수 없는 기능이면 그 사실을 밝히고, 지금 할 수 있는 준비를 쓰세요. 억지로 연결하지 마세요."
)
SYS_JUDGE = (
    "요약이 원문에서 뒷받침되는지 판정하세요. 헤드라인과 요약만 판단합니다.\n"
    "번역·단위 환산(예: $60 million → 6000만 달러, $15 billion → 150억 달러)은 문제가 아닙니다.\n"
    "숫자가 다르다고 판단하기 전에 million=백만, billion=10억 으로 직접 환산해 확인하세요.\n"
    "원문에 없는 사실, 틀린 숫자, 과장만 problems 에 적으세요."
)


class LLM:
    def __init__(self, offline: bool):
        self.offline = offline or not os.environ.get("OPENAI_API_KEY")
        self.calls = 0
        if not self.offline:
            from openai import OpenAI
            self.client = OpenAI(default_headers={"Accept-Encoding": "gzip, deflate"})

    @property
    def mode(self) -> str:
        return "offline" if self.offline else "llm"

    def _parse(self, system: str, user: str, schema):
        self.calls += 1
        return self.client.chat.completions.parse(
            model=NL["model"], temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=schema).choices[0].message.parsed

    def shortlist(self, items: list, n: int, with_summary: bool = False) -> list[Pick]:
        if self.offline:
            scored = sorted(range(len(items)), key=lambda i: -_kw_hits(items[i]["title"]))
            return [Pick(index=i, reason=f"키워드 {_kw_hits(items[i]['title'])}개 일치 (offline 규칙)",
                         criterion=0, event=_event_key(items[i]["title"])) for i in scored[:n]]
        listing = "\n".join(
            f"{i}. [{it['source']}] {it['title']}"
            + (f"\n   요약: {it['summary'][:200]}" if with_summary and it["summary"] else "")
            for i, it in enumerate(items))
        sys_ = (f"{CRITERIA}\n\n후보를 서로 비교해 중요한 순서대로 {n}건을 고르세요. '버릴 것'은 고르지 마세요.\n"
                "같은 사건을 다룬 기사에는 반드시 같은 event 라벨을 붙이세요.")
        out = self._parse(sys_, listing, Shortlist)
        return [p for p in out.picks if 0 <= p.index < len(items)]   # 없는 번호는 버린다

    def draft(self, item: dict, body: str, feedback: list[str] | None = None) -> Draft:
        if self.offline:
            sents = re.split(r"(?<=[.!?다])\s+", body.strip())
            return Draft(headline=item["title"][:40], summary=" ".join(sents[:3])[:400],
                         insight="(offline 모드 — 인사이트는 LLM 실행 때 생성됩니다)", topic="-")
        user = f"[제목]\n{item['title']}\n\n[본문]\n{body[:6000]}"
        if feedback:
            user += "\n\n[이전 요약의 문제 — 이번에는 고치세요]\n" + "\n".join(f"- {p}" for p in feedback)
        return self._parse(SYS_DRAFT, user, Draft)

    def judge(self, body: str, d: dict, hint: list[str]) -> Verdict:
        if self.offline:
            return Verdict(ok=not hint, problems=[f"원문에 없는 숫자: {h}" for h in hint])
        user = f"[원문]\n{body[:5000]}\n\n[헤드라인]\n{d['headline']}\n\n[요약]\n{d['summary']}"
        if hint:
            user += "\n\n[참고] 원문에서 글자 그대로 찾지 못한 숫자: " + ", ".join(hint)
        return self._parse(SYS_JUDGE, user, Verdict)


def _kw_hits(title: str) -> int:
    t = title.lower()
    return sum(k.lower() in t for k in AUD["keywords"])


def _event_key(title: str) -> str:
    words = re.findall(r"[0-9A-Za-z가-힣]{2,}", title)
    return " ".join(words[:3])


def missing_numbers(text: str, body: str) -> list[str]:
    """요약 속 숫자 중 원문에 글자 그대로 없는 것. 환산된 숫자도 걸리므로 판정이 아니라 '힌트'로만 쓴다."""
    flat = body.replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return sorted({n for n in nums if len(n) >= 2 and n not in flat})


_llm: LLM | None = None


def llm() -> LLM:
    assert _llm is not None, "run() 으로 실행하세요"
    return _llm


# ── ① 수집 ───────────────────────────────────────────────────
def collect(s: dict) -> dict:
    now = now_utc()
    cutoff = now - timedelta(hours=s["hours"])
    items, dead, seen_url, seen_title = [], [], set(), set()
    per_source, fixed_tz = {}, 0
    for src in [x for x in CONFIG["sources"] if x["adopted"]]:
        try:
            r = requests.get(src["url"], headers=UA, timeout=20)
            r.raise_for_status()
            entries = feedparser.parse(r.content).entries
        except Exception as e:                      # 한 곳이 죽어도 나머지는 계속
            dead.append(f"{src['name']}({type(e).__name__})")
            continue
        n = 0
        for e in entries:
            at, fixed = entry_time(e, now)
            if not at or at < cutoff:
                continue
            u, t = norm_url(e.link), re.sub(r"\W+", "", e.get("title", "")).lower()
            if u in seen_url or t in seen_title:     # 같은 글이 두 번 들어오는 것을 막는다
                continue
            seen_url.add(u)
            seen_title.add(t)
            fixed_tz += fixed
            n += 1
            items.append({"title": strip_tags(e.get("title", "")), "url": clean_url(e.link), "source": src["name"],
                          "at": at.isoformat(), "summary": strip_tags(e.get("summary", ""))[:300]})
        per_source[src["name"]] = n
    line = f"① 수집   {s['hours']}시간 · {len(items)}건 {per_source}"
    if fixed_tz:
        line += f" · 시각 보정 {fixed_tz}건"
    if dead:
        line += f" · 응답 없음 {dead}"                # 조용한 실패를 막는다
    return {"collected": items, "log": [line],
            "stats": [{"stage": "collect", "count": len(items), "per_source": per_source,
                       "dead": dead, "tz_fixed": fixed_tz}]}


# ── ② 선별 ───────────────────────────────────────────────────
def select(s: dict) -> dict:
    items, batch, keep = s["collected"], NL["batch"], NL["survivors_per_batch"]
    survivors = []
    for i in range(0, len(items), batch):          # 예선 — 묶음끼리는 서로 모른 채
        chunk = items[i:i + batch]
        survivors += [chunk[p.index] for p in llm().shortlist(chunk, min(keep, len(chunk)))]
    finals, labels, events = [], [], set()
    if survivors:                                   # 본선 — 한 화면에 놓고 비교
        # 본선은 요약까지 보고 판단한다 — '해외 한정' 같은 조건은 제목에 없다
        for p in llm().shortlist(survivors, min(NL["target"] * 2, len(survivors)), with_summary=True):
            it = survivors[p.index]
            dup = p.event.strip().lower() in events
            capped = sum(f["source"] == it["source"] for f in finals) >= NL["max_per_source"]
            kept = not dup and not capped and len(finals) < NL["target"]
            labels.append({"title": it["title"], "source": it["source"], "event": p.event,
                           "criterion": p.criterion, "reason": p.reason, "kept": kept,
                           "why_not": None if kept else ("같은 사건" if dup else "소스 상한" if capped else "순위 밖")})
            if not kept:
                continue
            events.add(p.event.strip().lower())
            finals.append({**it, "event": p.event, "reason": p.reason, "criterion": p.criterion})
    log = [f"② 선별   {len(items)} → 예선 {len(survivors)} → 본선 {len(finals)}건"]
    for x in labels:
        mark = "✓" if x["kept"] else "·"
        tail = "" if x["kept"] else f" ({x['why_not']})"
        log.append(f"   {mark} [{x['event']}] 기준{x['criterion']} {x['source']} · {x['title'][:38]}{tail}")
    return {"picked": finals, "log": log,
            "stats": [{"stage": "select", "in": len(items), "prelim": len(survivors),
                       "final": len(finals), "labels": labels}]}


# ── ③ 취재 (기사마다 워커) ────────────────────────────────────
class ReportIn(TypedDict):
    item: dict


def fan_report(s: dict):
    if not s["picked"]:                 # 고른 게 없으면 검수·발행으로 바로 — 조용한 날에도 안내는 나간다
        return "verify"
    return [Send("report", {"item": it}) for it in s["picked"]]


def report(s: ReportIn) -> dict:
    it = s["item"]
    body = extract_body(it["url"])
    if not body or len(body) < MIN_BODY:
        return {"log": [f"③ 취재   제외 · {it['source']} 본문 {len(body or '')}자"],
                "stats": [{"stage": "report", "title": it["title"], "skipped": "short_body"}]}
    d = llm().draft(it, body)
    return {"drafted": [{**it, "body": body[:6000], **d.model_dump()}],
            "log": [f"③ 취재   {it['source']} · {d.headline}"]}


# ── ④ 검수 (실패하면 한 번 다시 쓰고, 또 실패하면 뺀다) ──────────
def verify(s: dict) -> dict:
    kept, log, results = [], [], []
    for d in s["drafted"]:
        hint = missing_numbers(d["headline"] + " " + d["summary"], d["body"])
        v = llm().judge(d["body"], d, hint)
        outcome = "pass"
        if not v.ok:                                  # 대안 1: 지적을 붙여 재생성
            nd = llm().draft(d, d["body"], v.problems)
            d2 = {**d, **nd.model_dump()}
            hint2 = missing_numbers(d2["headline"] + " " + d2["summary"], d2["body"])
            v2 = llm().judge(d2["body"], d2, hint2)
            if v2.ok:
                d, outcome = d2, "pass_after_retry"
            else:                                     # 대안 2: 스킵 — 틀린 요약은 내보내지 않는다
                outcome = "dropped"
                v = v2
        results.append({"title": d["title"], "outcome": outcome, "problems": v.problems, "num_hint": hint})
        if outcome != "dropped":
            kept.append(d)
        log.append(f"   {'✓' if outcome != 'dropped' else '✗'} {outcome:<16} {d['headline'][:36]}"
                   + (f" · {v.problems[:1]}" if v.problems else ""))
    head = f"④ 검수   {len(s['drafted'])} → {len(kept)}건"
    if len(kept) < NL["min_publish"]:
        head += f" · ⚠ 최소 {NL['min_publish']}건 미달"
    return {"verified": kept, "log": [head, *log], "stats": [{"stage": "verify", "results": results}]}


# ── ⑤ 발행 ───────────────────────────────────────────────────
def render_md(day: str, arts: list, mode: str) -> str:
    out = [f"# {NL['name']} · {day}", ""]
    if mode == "offline":
        out += ["> ⚠ offline 모드로 만든 브리핑입니다 (LLM 미사용 — 배선 점검용).", ""]
    if not arts:
        return "\n".join(out + ["오늘은 조용합니다. 독자 기준에 맞는 새 소식이 없었습니다.", ""])
    out += [f"오늘은 {len(arts)}건을 골랐습니다. ({', '.join(dict.fromkeys(a['source'] for a in arts))})", ""]
    for i, a in enumerate(arts, 1):
        when = datetime.fromisoformat(a["at"]).astimezone(KST).strftime("%m-%d %H:%M")
        out += [f"## {i}. {a['headline']}", "", a["summary"], "",
                f"**💡 이번 주 해 볼 것:** {a['insight']}", "",
                f"<sub>{a['topic']} · {a['source']} · {when} KST · [원문]({a['url']})</sub>", ""]
    return "\n".join(out)


def render_html(md_title: str, arts: list, mode: str) -> str:
    cards = "".join(
        f"<article><h2>{i}. {html.escape(a['headline'])}</h2><p>{html.escape(a['summary'])}</p>"
        f"<p class=tip>💡 {html.escape(a['insight'])}</p>"
        f"<p class=meta>{html.escape(a['topic'])} · {html.escape(a['source'])} · "
        f"<a href=\"{html.escape(a['url'])}\">원문</a></p></article>"
        for i, a in enumerate(arts, 1)) or "<p>오늘은 조용합니다.</p>"
    warn = "<p class=warn>⚠ offline 모드 (LLM 미사용)</p>" if mode == "offline" else ""
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(md_title)}</title><style>body{{font-family:system-ui,sans-serif;max-width:720px;"
            "margin:24px auto;padding:0 16px;background:#faf8f4;color:#222}article{background:#fff;border-radius:12px;"
            "padding:16px 20px;margin:14px 0;box-shadow:0 1px 3px #0001}.tip{background:#fff6e0;padding:8px 12px;"
            "border-radius:8px}.meta{color:#777;font-size:13px}.warn{color:#b45309}</style>"
            f"<h1>{html.escape(md_title)}</h1>{warn}{cards}")


def publish(s: dict) -> dict:
    arts, day = s["verified"], datetime.now(KST).strftime("%Y-%m-%d")
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    md = render_md(day, arts, s["mode"])
    md_path, html_path = out_dir / f"{day}.md", out_dir / f"{day}.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(render_html(f"{NL['name']} · {day}", arts, s["mode"]), encoding="utf-8")
    back = md_path.read_text(encoding="utf-8")       # 발행 확인 — 다시 읽어 헤드라인이 다 있는지
    missing = [a["headline"] for a in arts if a["headline"] not in back]
    ok = not missing
    label = f"{len(arts)}건" if arts else "조용합니다(안내 발행)"
    return {"log": [f"⑤ 발행   {label} → out/{md_path.name}, out/{html_path.name} · 확인 {'OK' if ok else f'누락 {missing}'}"],
            "stats": [{"stage": "publish", "count": len(arts), "file": md_path.name, "confirmed": ok}]}


# ── 그래프와 실행 ─────────────────────────────────────────────
def build():
    g = StateGraph(Brief)
    for name in ("collect", "select", "report", "verify", "publish"):
        g.add_node(name, globals()[name])
    g.add_edge(START, "collect")
    g.add_edge("collect", "select")
    g.add_conditional_edges("select", fan_report, ["report", "verify"])   # 기사 수만큼 워커를 펼친다
    g.add_edge("report", "verify")
    g.add_edge("verify", "publish")
    g.add_edge("publish", END)
    return g.compile()


def run(hours: int | None = None, offline: bool = False) -> dict:
    global _llm
    _llm = LLM(offline)
    t0 = time.time()
    init = {"hours": NL["hours"] if hours is None else hours, "mode": _llm.mode, "collected": [], "picked": [],
            "drafted": [], "verified": [], "stats": [], "log": [f"모드: {_llm.mode} · 모델 {NL['model']}"]}
    out = build().invoke(init)
    rec = {"run_at": datetime.now(KST).isoformat(timespec="seconds"), "mode": _llm.mode,
           "model": NL["model"], "hours": init["hours"], "llm_calls": _llm.calls,
           "seconds": round(time.time() - t0, 1), "stages": out["stats"]}
    store = ROOT / "store"
    store.mkdir(exist_ok=True)
    with (store / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out["log"].append(f"기록   store/metrics.jsonl · LLM 호출 {_llm.calls}회 · {rec['seconds']}초")
    return out


def mermaid() -> str:
    return build().get_graph().draw_mermaid()
