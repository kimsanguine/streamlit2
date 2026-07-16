# 🔍 심화(선택) — 완성형: 스트리밍 tool-calling 버전 (빈칸 없음 — 읽고 실행만 해보면 된다)
# [왜] 본 실습(apps/m5b_agent_loop.py)은 비스트리밍으로 think-act-observe를 가르쳤다 — Lab 1의 목적은
#      스트리밍 없이도 100% 달성된다. 이 파일은 도구 정의·판단·실행 로직은 완전히 같고,
#      "LLM 응답을 어떻게 받는가"만 다른 스트리밍 버전이다. 실무 챗봇은 체감 속도를 위해 이 방식을 자주 쓴다.
# [핵심 차이] stream=True로 받으면 tool_calls가 여러 청크에 걸쳐 조각조각 온다(이름/인자가 나뉘어 도착).
#      apps/m5b_agent_loop.py처럼 response.choices[0].message.tool_calls를 한 번에 읽을 수 없어서,
#      tc.index를 키 삼아 slot["name"] += ..., slot["arguments"] += ...로 직접 이어 붙여야 완전한 값이 된다.
#      아래 run_agent의 for chunk in stream: 블록이 그 누적 로직이다.
# [재사용] TOOLS·run_tool·PROVIDERS·provider_available·get_client는 apps/m5b_agent_loop.py와 완전히
#      같다 — 도구 자체는 스트리밍이든 아니든 똑같이 동작하기 때문에 그대로 import한다.
# 실행: python3.11 -m streamlit run apps/m5b_agent_loop_stream.py

import json
import re

import streamlit as st

# [왜] `streamlit run apps/파일.py`는 apps/ 폴더만 import 경로에 넣는다 — `from apps...`가 되려면
#      프로젝트 루트(이 파일의 상위 폴더)를 경로에 직접 추가해야 어떤 실행 방식·위치에서도 동작한다.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps.m5b_agent_loop import PROVIDERS, TOOLS, get_client, provider_available, run_tool


# ── 스트리밍 think-act-observe 루프 — 조각을 이어 붙여야 완전한 tool_calls가 된다 ──
def run_agent(question: str, provider: str, max_turns: int = 5):
    """apps/m5b_agent_loop.py의 run_agent와 판단·행동·관찰 흐름은 동일 — 응답을 stream=True로 받는 점만 다르다."""
    if not provider_available(provider):
        yield from _fake_route(question)
        return
    try:
        client = get_client(provider)
        model = PROVIDERS[provider]["model"]
    except Exception as e:  # noqa: BLE001
        yield f"⚠️ LLM 클라이언트 초기화 실패: {e}"
        return
    messages = [
        {"role": "system", "content": "너는 감성분석·계산 도구를 쓸 수 있는 한국어 조수다. 필요하면 도구를 호출하고 그 결과로 답해라."},
        {"role": "user", "content": question},
    ]
    for _ in range(max_turns):
        try:
            stream = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, stream=True)
            content, tool_calls = [], {}
            for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content.append(delta.content)
                    yield delta.content  # 최종 답변 실 토큰 스트리밍
                for tc in getattr(delta, "tool_calls", None) or []:
                    # [핵심] 도구 이름·인자가 여러 청크로 쪼개져 온다 — index를 키로 이어 붙인다(누적).
                    slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except Exception as e:  # noqa: BLE001
            yield f"\n\n⚠️ LLM 호출 실패: {e}"
            return

        if not tool_calls:
            return  # 도구가 더 필요 없는 턴 = 최종 답변(이미 스트리밍됨)
        calls = [tool_calls[i] for i in sorted(tool_calls)]
        messages.append({"role": "assistant", "content": "".join(content) or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(c["name"], args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            yield f"🔧 `{c['name']}` 호출 → {result}\n\n"
    yield "\n\n⚠️ 최대 턴 수에 도달해 종료합니다. 질문을 더 단순하게 나눠보세요."


def _fake_route(question: str):
    """키 없이 동작하는 데모 — 키워드로 도구를 고른다(실 LLM은 같은 도구를 스스로 고른다)."""
    yield "🧪 **데모 모드**(LLM 키 없음) — 키워드로 도구를 고릅니다. 키를 넣으면 LLM이 스스로 고릅니다.\n\n"
    if any(k in question for k in ["감성", "감정", "리뷰", "기분"]):
        text = question.split(":", 1)[1].strip() if ":" in question else question
        yield f"🔧 `analyze_sentiment` 호출\n\n{run_tool('analyze_sentiment', {'text': text})}\n"
    elif any(c.isdigit() for c in question):
        expr = "".join(re.findall(r"[0-9+\-*/.() ]", question)).strip()
        yield f"🔧 `calculate` 호출\n\n{run_tool('calculate', {'expression': expr})}\n"
    else:
        yield "감성분석 또는 계산 질문을 해보세요. 예) `감성분석: 이 영화 최고예요` / `150 * 8500`\n"


def main():
    st.set_page_config(page_title="맨손 에이전트 루프 (스트리밍)", page_icon="🤖", layout="centered")
    st.title("🤖 맨손 에이전트 루프 — 스트리밍 버전 (심화)")
    st.caption("apps/m5b_agent_loop.py(비스트리밍)와 도구·판단 로직은 같고, 응답을 받는 방식만 다르다")

    provider_names = list(PROVIDERS)
    provider = st.sidebar.selectbox("Provider", provider_names, index=provider_names.index("openai"))
    if not provider_available(provider):
        st.info("🧪 데모 모드 — provider 키/서버가 없어 키워드 라우터로 동작합니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("예) 감성분석: 이 영화 최고예요  /  150 * 8500"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            answer = st.write_stream(run_agent(prompt, provider))
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
