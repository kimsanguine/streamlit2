import streamlit as st

from showcase.core.models import predict_sentiment
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="감성 분석", page_icon="💬", layout="wide")
apply_theme()
theme_toggle()
hero("💬 한국어 감성 분석", "KoELECTRA-small(NSMC 파인튜닝)로 문장의 긍정/부정을 판정합니다")

# [구성] 좌 입력 / 우 결과 2단 앵커링.
left, right = st.columns([5, 4], gap="large")
with left:
    with st.form("sentiment_form"):
        st.markdown("#### 문장 입력")
        text = st.text_area(
            "분석할 문장",
            value="이 영화 정말 재밌어요! 최고예요",
            height=140,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("분석하기", type="primary", use_container_width=True)

# [Codex] 결과를 입력 텍스트와 함께 저장하고 현재 입력과 일치할 때만 표시(페이지 복귀·빈 입력 stale 방지).
current = text.strip()
if submitted:
    if not current:
        st.warning("분석할 문장을 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            st.session_state["sentiment_out"] = {**predict_sentiment(current), "_text": current}

with right:
    out = st.session_state.get("sentiment_out")
    with st.container(border=True):
        if not out or out.get("_text") != current:
            st.markdown("### 결과 대기 중")
            st.caption("왼쪽에 문장을 넣고 **분석하기**를 누르면 여기에 판정이 나타납니다.")
        else:
            icon = "😊" if out["label"] == "긍정" else "😞"
            st.markdown(f"### {icon} {out['label']}")
            st.metric("확신도 (softmax 점수)", f"{out['score']:.1%}")
            st.progress(out["score"])
            # T3(미보정): m3와 정직성 통일 — 확신도를 보정 확률처럼 오해하지 않게.
            st.caption("확신도는 보정된 확률이 아닙니다 — 중립적인 문장도 90%+로 나올 수 있습니다.")
