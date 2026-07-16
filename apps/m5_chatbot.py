# 에이전트 챗봇 UI 4요소 — API 키 없이 오프라인으로 전체 흐름을 체감하는 데모
import time
import streamlit as st

st.set_page_config(page_title="에이전트 챗봇 UI", page_icon="🤖", layout="centered")
st.title("🤖 에이전트 챗봇 UI 4요소")
st.caption("API 키 불필요 — fake_llm_stream으로 실제 LLM 스트리밍을 흉내냅니다")

with st.expander("💡 이전 챕터와 연결"):
    st.markdown(
        "- **17·20강의 `messages` 리스트**: 아래 `st.session_state.messages`는 `role`/`content` "
        "딕셔너리 리스트로, 17강 ReAct 루프·20강 `qa_agent()`가 LLM에 보내던 messages와 완전히 같은 구조입니다.\n"
        "- **ST2에서 배운 `session_state` 관용구**: `if \"messages\" not in st.session_state:` — ST2에서 배운 "
        "최초 1회 초기화 패턴을 그대로 재사용합니다."
    )

# ── ③ session_state.messages — ST2 관용구 그대로 재사용 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# [왜] 실제 LLM 호출 없이도 UI 흐름 전체를 체감하도록 만든 가짜 스트리밍 함수
# [실제] 여기 자리에 client.chat.completions.create(..., stream=True) 응답을 그대로 넣으면 실제 LLM 연결
def fake_llm_stream(user_text: str):
    reply = f"'{user_text}'에 대한 답변입니다. 실제 서비스라면 여기서 검색 결과나 도구 실행 결과를 근거로 답합니다."
    for word in reply.split():
        yield word + " "
        time.sleep(0.05)  # [흐름] 단어 하나당 0.05초 — 토큰이 흘러나오는 느낌을 흉내냄

# ── ① 지난 대화 이력을 chat_message로 다시 그리기 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  # [통일] 메시지 표시는 st.markdown — 서식 지원 + 2~5절(m5b_agent_loop 등)과 일관

# ── ② chat_input — None이 아닐 때만 처리 ──
if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # [핵심] st.status로 에이전트의 작업 과정을 단계별로 보여줌 — 사용자가 "멈춘 건지 일하는 건지" 불안하지 않게
        with st.status("답변을 준비하는 중...", expanded=True) as status:
            st.write("1) 의도 분석 중...")
            time.sleep(0.3)
            st.write("2) 도구 호출 중...")
            time.sleep(0.3)
            st.write("3) 응답 생성 중...")
            status.update(label="완료", state="complete", expanded=False)

        # ── ④ write_stream — 제너레이터를 흘려보내고, 반환값은 전체 문자열 ──
        full_response = st.write_stream(fake_llm_stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": full_response})
