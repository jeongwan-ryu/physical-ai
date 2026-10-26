"""Agentic SOC: GuardDuty findings (via aws-security MCP) -> Gemini -> Korean incident report.

Usage:
    export GEMINI_API_KEY=...            # + AWS credentials (env vars or AWS_PROFILE)
    python soc_agent.py                  # 한 번 점검 후 종료
    python soc_agent.py --watch 300      # 300초마다 새 finding만 보고
    python soc_agent.py --severity HIGH  # LOW | MEDIUM | HIGH | ALL
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE / "silver-sniffle"
REPORT_DIR = HERE / "reports"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

SYSTEM_PROMPT = """너는 AWS 보안관제센터(SOC)의 Tier-2 분석가다.
GuardDuty finding을 받으면 aws-security MCP 도구로 필요한 만큼 추가 조사한다
(예: guardduty_security_operations의 get_finding_details, 관련 EC2/IAM/S3 리소스 조회).
도구는 조회 전용으로만 사용하고, 리소스를 변경하지 않는다.
도구 호출 시 추가 인자는 params에 JSON 문자열로 넣는다. 예: params='{"detector_id": "abc"}'

조사가 끝나면 아래 형식의 한국어 Markdown 보고서만 출력한다.

# GuardDuty 보안 사고 보고서
## 1. 요약 (심각도, 한 문단)
## 2. 탐지 신호 목록 (표: 유형 | 심각도 | 리소스 | 최초/최근 탐지 | 횟수)
## 3. 상황 분석 (무슨 일이 일어나고 있는지, 공격자 행위, MITRE ATT&CK 매핑, 오탐 가능성)
## 4. 영향 범위 (계정, 리전, 리소스, 노출 데이터)
## 5. 대응 방법
### 즉시 조치 (격리/차단, 실행 가능한 AWS CLI 명령 포함)
### 단기 조치 (조사·복구)
### 장기 조치 (재발 방지)
## 6. 추가 확인 필요 사항
근거 없는 내용은 추측이라고 명시한다."""


def call(session, tool, operation, **params):
    return session.call_tool(tool, {"operation": operation, "params": json.dumps(params)})


def text_of(result) -> dict:
    return json.loads(result.content[0].text) if result.content else {}


def new_findings(listing: dict, seen: set) -> list:
    """list_findings 결과에서 아직 보고하지 않은 finding만 골라 seen에 추가."""
    fresh = [f for f in listing.get("findings", []) if f.get("id") and f["id"] not in seen]
    seen.update(f["id"] for f in fresh)
    return fresh


async def detect(session, severity, seen) -> list | None:
    detectors = text_of(await call(session, "guardduty_security_operations", "list_detectors")).get("detectors", [])
    if not detectors:
        return None
    found = []
    for d in detectors:
        listing = text_of(await call(session, "guardduty_security_operations", "list_findings",
                                     detector_id=d["detector_id"], severity=severity, max_results=50))
        if "error" in listing:
            print("list_findings 오류:", listing["error"], file=sys.stderr)
            continue
        found += new_findings(listing, seen)
    return found


async def report(client, session, findings) -> Path:
    prompt = ("다음 GuardDuty finding들을 조사하고 보고서를 작성하라.\n```json\n"
              + json.dumps(findings, ensure_ascii=False, indent=2) + "\n```")
    resp = await client.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            tools=[session],  # Gemini가 MCP 도구를 직접 호출 (automatic function calling)
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(maximum_remote_calls=15),
        ),
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"soc_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(resp.text or "(Gemini 응답 없음)", encoding="utf-8")
    return path


async def main(args):
    client = genai.Client()  # GEMINI_API_KEY 환경변수 사용
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_DIR / "aws_security_mcp" / "main.py"), "stdio"],
        cwd=str(SERVER_DIR),
        env={**os.environ, "PYTHONPATH": str(SERVER_DIR)},
    )
    seen = set()  # ponytail: 메모리에만 저장, 재시작하면 기존 finding도 다시 보고됨. 필요하면 파일로 저장
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        if "guardduty_security_operations" not in tools:
            return "MCP 서버에 GuardDuty 도구가 없습니다. AWS 자격증명을 확인하세요."
        while True:
            findings = await detect(session, args.severity, seen)
            if findings is None:
                # MCP 서버는 AccessDenied도 빈 목록으로 돌려줌
                return ("GuardDuty detector를 찾지 못했습니다. 확인: (1) IAM 권한(guardduty:List*/Get*, SecurityAudit) "
                        "(2) 이 리전에서 GuardDuty 활성화 여부 (3) config.yaml의 region")
            stamp = f"{datetime.now():%H:%M:%S}"
            if findings:
                print(f"[{stamp}] 신규 finding {len(findings)}건 -> Gemini 분석 중...")
                print(f"[{stamp}] 보고서: {await report(client, session, findings)}")
            else:
                print(f"[{stamp}] 신규 finding 없음")
            if not args.watch:
                break
            await asyncio.sleep(args.watch)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--severity", default="ALL", choices=["LOW", "MEDIUM", "HIGH", "ALL"])
    p.add_argument("--watch", type=int, default=0, metavar="SEC", help="폴링 주기(초), 0이면 1회 실행")
    sys.exit(asyncio.run(main(p.parse_args())))
