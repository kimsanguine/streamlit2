# showcase/core/agent.py — 20강 tool-calling 루프를 계승한 에이전트 + 키 없이 동작하는 fake 폴백.
# [규약] OpenAI 호환 provider만 사용(base_url 전환) — Anthropic/Claude 금지(20강 동일).
# [핵심] 배포 URL은 키가 없어도 동작해야 하므로, provider 불가 시 fake_route(결정적 키워드 라우팅)로 폴백.
import json
import logging
import os
import re
from typing import Iterator

import streamlit as st

from showcase.core.tools import AGENT_TOOLS, run_tool

# 20강 강의_AI_Pair/agent_app/config.py와 동일한 base_url·model. 키만 st.secrets→env로 로딩(Cloud Secrets 대응).
PROVIDERS = {
    "local": {"base_url": "http://localhost:11434/v1", "model": "hermes3:8b", "static_key": "ollama"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free", "key_env": "OPENROUTER_API_KEY"},
    "openai": {"base_url": None, "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},  # base_url=None → 공식 기본 주소
}

_SYSTEM = (
    "너는 3개의 ML 도구(classify_penguin: 펭귄 종 분류, analyze_sentiment: 한국어 감성분석, "
    "classify_image: 최근 업로드 이미지 분류)를 쓸 수 있는 한국어 조수다. "
    "질문에 도구가 필요하면 도구를 호출하고, 그 결과를 바탕으로 한국어로 간결히 답해라."
)


def _get_key(provider: str) -> str:
    cfg = PROVIDERS[provider]
    if "static_key" in cfg:
        return cfg["static_key"]
    env = cfg["key_env"]
    try:
        if env in st.secrets:  # 1) Streamlit Cloud Secrets
            return str(st.secrets[env])
    except Exception:
        pass
    return os.environ.get(env, "")  # 2) 환경변수


def provider_available(provider: str) -> bool:
    """provider가 실제로 호출 가능한지 예외 없이 bool로 반환. 배포 URL(키 없음)에선 False가 정상."""
    if provider not in PROVIDERS:
        return False
    if provider == "local":
        try:  # ollama 서버가 떠 있는지 짧게 확인
            import urllib.request

            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
            return True
        except Exception:
            return False
    return bool(_get_key(provider))


@st.cache_resource
def get_client(provider: str):
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    return OpenAI(base_url=cfg["base_url"], api_key=_get_key(provider) or "none")


def _real_agent(question: str, provider: str, session_image=None, max_turns: int = 5) -> Iterator[str]:
    client = get_client(provider)
    model = PROVIDERS[provider]["model"]
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]

    for _ in range(max_turns):
        # [M4+Codex#5] 매 턴 stream=True 한 번만 호출하고, 델타에서 content(실 토큰)와 tool_calls를 함께
        #   누적한다. 이렇게 하면 최종 답변은 '진짜 토큰 스트리밍'(M4)이면서, 비스트리밍 호출 뒤 stream=True를
        #   또 부르던 '이중 호출'(Codex#5: 비용·지연 2배 + 답 불일치)이 사라진다.
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=AGENT_TOOLS, stream=True
        )
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason = None
        for chunk in stream:
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield delta.content  # 최종 답변 실 토큰 스트리밍
            if getattr(delta, "refusal", None):  # [Codex] safety refusal 스트림 — 빈 답변으로 끝나지 않게 노출
                content_parts.append(delta.refusal)
                yield delta.refusal
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:  # [Codex-PLAUSIBLE] id는 tool_call당 첫 델타에 완결로 온다(arguments만 분할) → 덮어쓰기가 정상
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        if not tool_calls:
            # [Codex] 도구 없는 최종 턴 — 빈 응답/길이 초과를 사용자에게 알린다(빈 화면·잘린 답 방지).
            if not content_parts:
                yield "(모델이 빈 응답을 반환했습니다 — 다시 시도하거나 질문을 바꿔보세요.)"
            elif finish_reason == "length":
                yield "\n\n(⚠️ 응답이 최대 길이에서 잘렸습니다.)"
            return  # content가 이미 스트리밍됨 = 최종 답변
        calls = [tool_calls[i] for i in sorted(tool_calls)]
        messages.append({
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
        })
        for c in calls:
            args = json.loads(c["arguments"] or "{}")
            result = run_tool(c["name"], args, session_image=session_image)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            yield f"🔧 `{c['name']}` 호출 → {result}\n\n"
    yield "\n(도구 호출이 최대 턴 수에 도달했습니다.)"


def fake_route(question: str, session_image=None) -> Iterator[str]:
    """키 없이 동작하는 데모 라우터 — 키워드로 도구를 고르고 run_tool을 '실제로' 호출한다.
    [M4] 데모 모드도 어떤 tool을 왜 골랐는지 trace를 노출해, 실 LLM 연결과 같은 경험을 준다."""
    q = question
    yield "🧪 **데모 모드** (LLM 키 없음) — 키워드로 도구를 고릅니다. 실제 provider를 연결하면 **LLM이 같은 도구를 스스로 선택**합니다.\n\n"

    # [Codex] 구체적인 분기(펭귄=키워드+숫자4, 이미지=키워드)를 먼저 확인하고, 느슨한 감성 키워드는 마지막에.
    #   '부정확'이 '부정'을 부분 포함해 펭귄 질문이 감성으로 새던 것을 막는다(구체 분기 우선).
    nums = re.findall(r"[-+]?\d+\.?\d*", q)
    if ("펭귄" in q or "종 분류" in q) and len(nums) >= 4:
        # [Codex] 측정값은 설명 텍스트 뒤에 온다 — '펭귄 3종 분류: 39 18 181 3750'의 '3종' 같은 stray
        #   숫자에 밀리지 않도록 앞 4개가 아니라 '마지막 4개'를 측정값으로 쓴다.
        keys = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        args = dict(zip(keys, (float(n) for n in nums[-4:])))
        yield f"🔧 `classify_penguin` 호출 (args={args})\n\n"
        yield run_tool("classify_penguin", args, session_image=session_image) + "\n"
        return

    # [Codex] 이미지 '존재'만으로 분류하지 않는다 — 텍스트가 실제로 이미지 분류를 요구할 때만.
    #   (이전엔 `or session_image is not None`이라, 이미지 올린 뒤 "안녕하세요"에도 분류가 돌았다.)
    if "이미지" in q or "사진" in q:
        if session_image is None:
            yield "이미지가 업로드되지 않았습니다 — 먼저 사이드바에서 이미지를 올려주세요.\n"
            return
        yield "🔧 `classify_image` 호출 (최근 업로드 이미지)\n\n"
        yield run_tool("classify_image", {}, session_image=session_image) + "\n"
        return

    # [Codex] '긍정'/'부정'은 결과(outcome) 단어라 '부정확' 같은 단어에 오탐된다 — 요청/주제 단어만 트리거로.
    if any(k in q for k in ["감성", "감정", "리뷰", "기분"]):
        text = q.split(":", 1)[1].strip() if ":" in q else q
        yield f"🔧 `analyze_sentiment` 호출 (text={text!r})\n\n"
        yield run_tool("analyze_sentiment", {"text": text}, session_image=session_image) + "\n"
        return

    yield ("어떤 도구를 쓸지 못 정했어요. 예) `감성분석: 이 영화 최고예요` / "
           "`펭귄 종 분류: 39 18 181 3750` / 이미지를 올린 뒤 `이미지 분류해줘`.\n")


def run_agent(question: str, provider: str, session_image=None) -> Iterator[str]:
    """provider가 가능하면 실제 tool-calling 루프, 아니면(또는 시작 전 실패 시) fake_route로 자동 폴백."""
    if not provider_available(provider):
        yield from fake_route(question, session_image=session_image)
        return
    produced = False
    try:
        for piece in _real_agent(question, provider, session_image=session_image):
            produced = True
            yield piece
    except Exception as e:  # noqa: BLE001 — 실 호출 실패 처리
        logging.exception("real agent 실패")
        if produced:
            # [Codex] 이미 일부 출력(도구 호출 등)을 냈다면 fake_route로 전체 재실행하지 않는다 —
            #   같은 도구가 두 번 돌고 실경로/데모 결과가 섞이는 것 방지. 오류만 알린다.
            yield f"\n⚠️ 응답 도중 provider 오류가 발생했습니다({e})."
        else:
            # 아직 아무 출력도 안 냈으면(시작 전 실패) 데모 모드로 깨끗이 폴백(배포 URL 상시 응답).
            yield f"\n⚠️ provider 호출 실패({e}) — 데모 모드로 전환합니다.\n\n"
            yield from fake_route(question, session_image=session_image)
