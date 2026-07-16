import streamlit as st
from PIL import Image

from showcase.core.agent import PROVIDERS, provider_available, run_agent
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="에이전트", page_icon="🤖", layout="wide")
apply_theme()
theme_toggle()
hero(
    "🤖 tool-calling 에이전트",
    "펭귄·이미지·감성 3개 모델을 도구로 쓰는 에이전트 — 아무 문장이나 넣으면 어떤 도구를 왜 불렀는지 그대로 보입니다.",
)

provider_names = list(PROVIDERS.keys())
provider = st.sidebar.selectbox("Provider", provider_names, index=provider_names.index("openai"))
available = provider_available(provider)

# 이미지 업로더를 사이드바 expander로 정리 — 이미지 도구 쓸 때만 펼침(사이드바 밀림 방지).
with st.sidebar.expander("🖼️ 이미지 첨부 (이미지 분류 도구용)"):
    up = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up is not None:
        # [Codex] 손상/위장 이미지가 페이지를 죽이지 않게 예외 처리.
        try:
            st.session_state["agent_image"] = Image.open(up).convert("RGB")
            st.image(st.session_state["agent_image"], use_container_width=True)
        except Exception:  # noqa: BLE001
            st.session_state.pop("agent_image", None)
            st.error("이미지를 열 수 없습니다 — 손상되었거나 지원하지 않는 형식입니다.")
    else:
        # [Codex] 업로더를 비우면 이전 이미지를 제거 — 제거한 이미지를 다시 분류하는 것 방지.
        st.session_state.pop("agent_image", None)
session_image = st.session_state.get("agent_image")

if available:
    # [Codex] '사용 가능'은 키 존재(원격)·서버 응답(local)까지만 확인 — 키 유효성은 첫 요청에서 판명.
    st.success(
        f"✅ `{provider}` 사용 가능 — 실제 LLM이 도구를 선택합니다. "
        "(실제 연결은 첫 요청에서 확인되며, 인증 실패 시 데모 모드로 자동 전환됩니다.)"
    )
else:
    # [M4] 데모 모드를 결핍이 아니라 '설계 의도'로 프레이밍 — 기본 경험이 약해 보이지 않게.
    st.info(
        "✅ **데모 모드로 실행 중** — 키가 없어도 **실제 도구를 호출**합니다. "
        "provider 키를 연결하면 **LLM이 같은 도구를 스스로 선택**합니다. "
        "아래 대화에 **어떤 도구를 왜 호출했는지 trace가 그대로** 표시됩니다."
    )

# [구성] 예시 프롬프트를 placeholder에서 표면으로 — 1클릭으로 도구 호출을 보게.
EXAMPLES = {
    "🎬 감성 분석": "이 영화 정말 최고예요 감성 분석해줘",
    "🐧 펭귄 분류": "펭귄 종 분류: 부리 39 깊이 18 물갈퀴 181 체중 3750",
    "🖼️ 이미지 분류": "방금 올린 이미지 분류해줘",
}
picked = st.pills("예시로 시작하기 (또는 아래에 직접 입력)", list(EXAMPLES), selection_mode="single", key="agent_example")

placeholder = "예) 감성분석: 이 영화 최고예요  /  펭귄 종 분류: 39 18 181 3750  /  (이미지 첨부 후) 이미지 분류해줘"
typed = st.chat_input(placeholder)  # 위치와 무관하게 화면 하단에 고정 렌더

# [Codex] pill을 해제하면 dedup 상태를 초기화 — 같은 예시를 다시 실행할 수 있게.
if picked is None:
    st.session_state.pop("_agent_last_pick", None)

# 프롬프트 결정: 직접 입력 우선, 없으면 새로 선택한 예시 pill(1회성 — 재실행 방지).
prompt = None
if typed:
    prompt = typed
elif picked and st.session_state.get("_agent_last_pick") != picked:
    st.session_state["_agent_last_pick"] = picked
    prompt = EXAMPLES[picked]

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

for m in st.session_state.agent_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 빈 상태 — 덩그러니 빈 채팅 방지(처리할 프롬프트가 없을 때만).
if not st.session_state.agent_messages and not prompt:
    with st.container(border=True):
        st.markdown("#### 무엇을 물어볼 수 있나요?")
        st.caption(
            "펭귄 측정값 → 종 분류 · 한국어 문장 → 감성 판정 · (이미지 첨부 후) 이미지 → ImageNet 분류. "
            "위 예시 칩을 누르면 바로 시작합니다."
        )

if prompt:
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer = st.write_stream(run_agent(prompt, provider, session_image=session_image))
    st.session_state.agent_messages.append({"role": "assistant", "content": answer})
