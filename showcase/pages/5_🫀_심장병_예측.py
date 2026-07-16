import streamlit as st

from apps.heart_disease import LABELS
from showcase.core.models import predict_heart
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="심장병 예측", page_icon="🫀", layout="wide")
apply_theme()
theme_toggle()
hero("🫀 심장병 위험 예측", "UCI Heart Disease 1,025명으로 학습한 RandomForest — 13개 지표로 위험 확률을 추정합니다")
st.warning("⚠️ 교육용 데모입니다 — 실제 의료 진단·상담을 대체하지 않습니다.")

# [구성] 좌 입력 / 우 결과 2단 — 펭귄 페이지와 동일한 고정 앵커 패턴.
left, right = st.columns([5, 4], gap="large")
with left:
    with st.form("heart_form"):
        st.markdown("#### 환자 지표 입력")
        c1, c2 = st.columns(2)
        age = c1.slider("나이", 20, 90, 55)
        sex = c1.radio("성별", list(LABELS["sex"]), horizontal=True)
        cp = c1.selectbox("가슴통증 유형 (cp)", list(LABELS["cp"]))
        trestbps = c1.slider("안정 시 혈압", 90, 200, 130)
        chol = c1.slider("콜레스테롤", 120, 570, 240)
        fbs = c1.radio("공복 혈당 mg/dl", list(LABELS["fbs"]), horizontal=True)
        restecg = c2.selectbox("안정 시 심전도", list(LABELS["restecg"]))
        thalach = c2.slider("최대 심박수 (thalach)", 70, 210, 150)
        exang = c2.radio("운동 유발 협심증", list(LABELS["exang"]), horizontal=True)
        oldpeak = c2.slider("운동 후 ST 하강 (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = c2.selectbox("ST 기울기 (slope)", list(LABELS["slope"]))
        ca = c2.slider("주요 혈관 수 (ca)", 0, 4, 0)
        thal = c2.slider("탈륨 검사 코드 (thal)", 0, 3, 2)
        submitted = st.form_submit_button("위험도 예측", type="primary", use_container_width=True)

# [Codex] 결과를 '그 결과를 만든 입력'과 함께 저장하고 일치할 때만 표시 — 펭귄 페이지의 stale 방지 패턴 재사용.
current = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
if submitted:
    features = {
        "age": age, "sex": LABELS["sex"][sex], "cp": LABELS["cp"][cp], "trestbps": trestbps,
        "chol": chol, "fbs": LABELS["fbs"][fbs], "restecg": LABELS["restecg"][restecg],
        "thalach": thalach, "exang": LABELS["exang"][exang], "oldpeak": oldpeak,
        "slope": LABELS["slope"][slope], "ca": ca, "thal": thal,
    }
    st.session_state["heart_out"] = {**predict_heart(features), "_inputs": current}

with right:
    out = st.session_state.get("heart_out")
    if out and out.get("_inputs") == current:
        st.markdown("#### 예측 결과")
        st.metric("심장병 위험 확률", f"{out['proba']:.1%}")
        st.progress(out["proba"], text="모델 추정 위험도 (RandomForest)")
        st.caption(
            f"hold-out 정확도 {out['test_acc']:.1%} 기준 점추정 — 보정(calibration)되지 않은 확률입니다. "
            "cp·thalach·oldpeak를 바꿔보며 판정 경계를 관찰해 보세요."
        )
        st.page_link("pages/4_🤖_에이전트.py", label="🤖 에이전트에게 '62세 심장병 위험'이라고 물어보기 →")
    else:
        st.info("지표를 조정하고 **위험도 예측**을 누르면 결과가 여기에 표시됩니다.")
