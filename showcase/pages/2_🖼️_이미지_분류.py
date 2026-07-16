import hashlib

import streamlit as st
from PIL import Image

from showcase.core.models import predict_image
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="이미지 분류", page_icon="🖼️", layout="wide")
apply_theme()
theme_toggle()
hero("🖼️ 이미지 분류", "MobileNetV2로 ImageNet 1000종을 분류합니다 (CPU 추론 약 87ms)")

# [구성] 좌 업로드·미리보기 / 우 top-5 결과 2단.
image = None
left, right = st.columns([5, 4], gap="large")
with left:
    st.markdown("#### 이미지 업로드")
    uploaded = st.file_uploader(
        "이미지 업로드", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded is not None:
        # [Codex] 확장자 필터는 위장·손상 파일을 막지 못한다 — Image.open 예외를 잡아 앱 중단 방지.
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="입력 이미지", use_container_width=True)
        except Exception:  # noqa: BLE001 — 손상/위장 이미지가 페이지를 죽이지 않게
            st.error("이미지를 열 수 없습니다 — 파일이 손상되었거나 지원하지 않는 형식입니다.")

with right:
    with st.container(border=True):
        if image is None:
            st.markdown("### 결과 대기 중")
            st.caption("왼쪽에 이미지를 올리면 top-5 예측이 여기에 나타납니다.")
        else:
            # [Codex] 같은 업로드의 재추론 방지 — 파일 서명이 바뀔 때만 predict_image 실행.
            #   서명은 (name, size)가 아니라 내용 해시 — 다른 이미지가 같은 이름·크기일 때의 캐시 충돌 방지.
            sig = hashlib.md5(uploaded.getvalue()).hexdigest()
            if st.session_state.get("_img_sig") != sig:
                with st.spinner("분류 중..."):
                    st.session_state["_img_top5"] = predict_image(image)
                st.session_state["_img_sig"] = sig
            top5 = st.session_state["_img_top5"]
            top_name, top_prob = top5[0]
            st.markdown(f"### 🔎 {top_name}")
            st.metric("1위 확률", f"{top_prob:.1%}")
            # top-5를 st.progress 스택으로 — st.dataframe(canvas)은 CSS 다크 테밍이 안 돼
            # 흰 표로 남으므로, 테마를 따르는 progress 막대로 순위·확률을 보여준다.
            st.caption("top-5 예측")
            for name, prob in top5:
                st.progress(prob, text=f"{name} · {prob:.1%}")
