import streamlit as st

from showcase.core.models import predict_penguin
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="펭귄 분류", page_icon="🐧", layout="wide")
apply_theme()
theme_toggle()
hero("🐧 펭귄 종 분류", "부리·물갈퀴·체중으로 팔머펭귄 3종(Adelie·Chinstrap·Gentoo)을 분류합니다 (RandomForest)")

# [구성] 좌 입력 / 우 결과 2단 — 결과가 버튼 아래로 흐르지 않고 입력 옆에 고정 앵커로 뜬다.
left, right = st.columns([5, 4], gap="large")
with left:
    with st.form("penguin_form"):
        st.markdown("#### 측정값 입력")
        c1, c2 = st.columns(2)
        bill_length = c1.number_input("부리 길이 (mm)", 30.0, 60.0, 45.0)
        bill_depth = c1.number_input("부리 깊이 (mm)", 13.0, 22.0, 17.0)
        flipper = c2.number_input("물갈퀴 길이 (mm)", 170.0, 235.0, 200.0)
        mass = c2.number_input("체중 (g)", 2700.0, 6300.0, 4200.0)
        submitted = st.form_submit_button("분류하기", type="primary", use_container_width=True)

# [Codex] 결과를 '그 결과를 만든 입력'과 함께 저장하고, 현재 폼 입력과 일치할 때만 보여준다.
#   key= 만으론 1.59.1에서 페이지 이동 시 위젯이 기본값으로 리셋돼 stale 결과가 남았다 → 입력-결과 매칭으로 해소.
current = (bill_length, bill_depth, flipper, mass)
if submitted:
    st.session_state["penguin_out"] = {
        **predict_penguin(
            {
                "bill_length_mm": bill_length,
                "bill_depth_mm": bill_depth,
                "flipper_length_mm": flipper,
                "body_mass_g": mass,
            }
        ),
        "_inputs": current,
    }

with right:
    out = st.session_state.get("penguin_out")
    with st.container(border=True):
        if not out or out.get("_inputs") != current:
            st.markdown("### 결과 대기 중")
            st.caption("왼쪽에 측정값을 넣고 **분류하기**를 누르면 여기에 예측이 나타납니다.")
        else:
            st.markdown(f"### 🐧 {out['species']}")
            st.metric("확신도 (트리 투표 비율)", f"{out['proba']:.1%}")
            st.progress(out["proba"])
            st.caption("확신도는 보정된 확률이 아닙니다 — '97%면 97% 맞다'는 뜻은 아닙니다.")
