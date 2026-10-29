"""후보 소스 빠른 탐색: 응답·건수·최신글 나이"""
import sys, feedparser, requests
from datetime import datetime, timezone
UA = {"User-Agent": "Mozilla/5.0 (newsletter-agent-course)"}
now = datetime.now(timezone.utc)
for url in sys.argv[1:]:
    try:
        r = requests.get(url, headers=UA, timeout=20)
        f = feedparser.parse(r.content)
        n = len(f.entries)
        ages = []
        for e in f.entries:
            t = e.get("published_parsed") or e.get("updated_parsed")
            if t: ages.append((now - datetime(*t[:6], tzinfo=timezone.utc)).total_seconds()/3600)
        new24 = sum(a <= 24 for a in ages)
        print(f"{r.status_code} {n:>4}건 24h:{new24:>3} 최신:{(min(ages) if ages else -1):6.1f}h  {url}")
    except Exception as e:
        print(f"ERR {type(e).__name__} {url}")
