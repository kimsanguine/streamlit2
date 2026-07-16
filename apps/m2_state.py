# session_state 관용구 + 콜백 vs 반환값 + st.form 3규칙
import streamlit as st

st.set_page_config(page_title="상태 관리 실습", page_icon="🔁")
st.title("🔁 session_state 실습")

# ── ① 최초 1회 초기화 관용구 ──
# [핵심] not in 체크가 없으면 매 rerun마다 0으로 덮어써서 M1의 실패가 반복됩니다
if "count" not in st.session_state:
    st.session_state.count = 0

st.subheader("1. 기본 카운터")
if st.button("➕ 증가", key="btn_plain_increment"):
    st.session_state.count += 1
st.metric("현재 값", st.session_state.count)

st.divider()

# ── ② 콜백(on_click) — rerun이 '시작되기 전'에 실행됨 ──
# [왜] 콜백은 rerun 시작 전에 실행되므로, 콜백이 끝난 뒤의 rerun에서는 이미 갱신된 값이 화면에 그려짐
#      "그 자리에서 1회만 처리하면 되는 로직"이면 반환값을, "다음 rerun에도 결과를 계속 보여줘야 하면" 콜백+state를 씁니다
def increment_callback():
    st.session_state.count += 1

st.subheader("2. 콜백으로 증가")
st.button("➕ 콜백으로 증가", key="btn_callback_increment", on_click=increment_callback)
st.caption("버튼을 누르는 즉시 콜백이 실행되고, 그 결과가 반영된 화면이 그려집니다.")

st.divider()

# ── ③ st.form 3규칙 데모 ──
# [규칙 1] 폼은 반드시 st.form_submit_button을 포함해야 함
# [규칙 2] 폼 안에는 st.button·st.download_button을 넣지 않음(form_submit_button만 허용)
# [규칙 3] 폼 내부 위젯에는 on_click 콜백을 걸지 않음 — 오직 submit 버튼만 rerun을 트리거
st.subheader("3. st.form — 여러 입력을 한 번에 제출")
with st.form("profile_form"):
    name = st.text_input("이름")
    age = st.number_input("나이", 0, 120, 20)
    submitted = st.form_submit_button("제출")  # [규칙 1] 폼에는 반드시 이 버튼이 있어야 함

if submitted:
    st.success(f"제출 완료: {name}, {age}세")
    st.caption("폼 안의 text_input·number_input은 제출 전까지 rerun을 유발하지 않습니다 — "
               "상호의존적인 여러 입력을 한 번에 묶어 처리할 때 유용합니다.")

st.divider()
# [핵심] 이 버튼은 st.metric(위)보다 아래에 있어, 클릭 시 새 rerun에서 metric이 먼저 그려질 땐 아직 count가 안 바뀌었다
#        ("두 번 눌러야 반영" 증상). st.rerun()으로 한 번 더 돌려 최신값을 반영한다.
#        → 원칙: state 수정은 그 값을 표시하는 위젯보다 "먼저" 오는 코드 경로에서. 못 바꾸면 st.rerun()으로 보강.
if st.button("🔄 카운터 초기화", key="btn_reset"):
    st.session_state.count = 0
    st.rerun()