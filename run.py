"""파이프라인 실행.

  python run.py              # .env 의 OPENAI_API_KEY 로 실행
  python run.py --offline    # LLM 없이 배선만 점검
  python run.py --hours 48   # 수집 기간을 바꿔서
  python run.py --mermaid    # 그래프 구조만 출력
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from graph import mermaid, run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--hours", type=int)
    ap.add_argument("--mermaid", action="store_true")
    a = ap.parse_args()
    if a.mermaid:
        print(mermaid())
        sys.exit(0)
    for line in run(a.hours, a.offline)["log"]:
        print(line)
