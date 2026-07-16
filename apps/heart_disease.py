# 보너스 — Heart Disease 위험 예측 대시보드 (AI Python 2 미니프로젝트 5의 Streamlit 버전)
# [왜] AI Python 2에서 노트북으로 했던 "심장병 분류" 분석을, ST2(m3_penguins)에서 배운
#      "EDA 탭 + 예측 탭" 패턴 그대로 웹 앱으로 옮긴 예시다 — 미니프로젝트를 배포 가능한
#      포트폴리오로 바꾸는 가장 짧은 경로를 보여준다.
# [데이터] UCI Heart Disease (Cleveland, CC BY 4.0) — data/heart.csv 동봉(1,025행).
# 실행: python3.11 -m streamlit run apps/heart_disease.py

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# [왜] 한글 라벨이 든 matplotlib 그림은 폰트를 지정하지 않으면 □□로 깨진다 —
#      로컬(맥/윈도우)과 Cloud(fonts-nanum) 어디서든 있는 폰트를 순서대로 시도한다(m3와 동일 규약).
matplotlib.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "NanumGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# [왜] 숫자 코드(0/1/2…)만 보여주면 무슨 뜻인지 알 수 없다 — 화면 위젯에는 한국어 라벨을 쓰고,
#      모델에는 원래 코드를 넘긴다(라벨→코드 역변환은 dict 순서 이용).
LABELS = {
    "sex": {"여성": 0, "남성": 1},
    "cp": {"전형 협심증": 0, "비전형 협심증": 1, "비협심 통증": 2, "무증상": 3},
    "fbs": {"120 이하": 0, "120 초과": 1},
    "restecg": {"정상": 0, "ST-T 이상": 1, "좌심실 비대": 2},
    "exang": {"없음": 0, "있음": 1},
    "slope": {"상승": 0, "평탄": 1, "하강": 2},
}


@st.cache_data
def load_data() -> pd.DataFrame:
    # [왜] 앱 파일 기준 상대 경로 — 어디서 실행해도(로컬·Cloud) 같은 위치의 data/를 찾는다.
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "heart.csv")


@st.cache_resource
def train_model(df: pd.DataFrame):
    """RandomForest 분류기 학습. 반환: (model, test_acc, X_test 크기) — 재실행마다 재학습하지 않도록 캐싱."""
    X, y = df[FEATURES], df["target"]
    # [왜] stratify=y — 심장병/정상 비율을 train/test에 동일하게 유지한다(ST2에서 배운 층화).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, float(model.score(X_test, y_test)), len(X_test)


def main():
    st.set_page_config(page_title="심장병 위험 예측", page_icon="🫀", layout="wide")
    st.title("🫀 Heart Disease 위험 예측 대시보드")
    st.caption("UCI Heart Disease(Cleveland) 1,025명 — EDA와 RandomForest 예측을 한 화면에서.")
    # [왜] 의료 주제는 오해가 위험하다 — 교육용임을 첫 화면에서 분명히 밝힌다.
    st.warning("⚠️ 교육용 데모입니다 — 실제 의료 진단·상담을 대체하지 않습니다.")

    df = load_data()
    model, test_acc, n_test = train_model(df)

    tab_eda, tab_pred = st.tabs(["📊 데이터 탐색 (EDA)", "🩺 위험 예측"])

    with tab_eda:
        c1, c2, c3 = st.columns(3)
        c1.metric("표본 수", f"{len(df):,}명")
        c2.metric("심장병 비율", f"{df['target'].mean():.1%}")
        c3.metric("Test 정확도", f"{test_acc:.1%}", help=f"hold-out {n_test}명 기준 점추정 — 분할에 따라 출렁일 수 있습니다.")

        left, right = st.columns(2)
        with left:
            # [왜] 연령×타깃 겹친 히스토그램 — "심장병 그룹이 어느 연령대에 몰리는지"가 한눈에 보인다.
            fig, ax = plt.subplots(figsize=(6, 4))
            for t, (label, color) in enumerate([("정상", "#4C9BE8"), ("심장병", "#E8604C")]):
                ax.hist(df.loc[df["target"] == t, "age"], bins=20, alpha=0.6, label=label, color=color)
            ax.set_xlabel("나이")
            ax.set_ylabel("인원")
            ax.set_title("연령 분포 — 정상 vs 심장병")
            ax.legend()
            st.pyplot(fig)
        with right:
            # [왜] 타깃과의 상관 상위 피처 — "모델이 무엇을 볼 만한가"를 EDA 단계에서 먼저 가늠한다.
            corr = df.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            corr.head(8).iloc[::-1].plot.barh(ax=ax2, color="#7C6CE8")
            ax2.set_title("타깃과의 절대 상관 상위 8개 피처")
            ax2.set_xlabel("|상관계수|")
            st.pyplot(fig2)
        st.caption("cp(가슴통증 유형)·thalach(최대 심박수)·exang(운동 유발 협심증)이 상위권 — 예측 탭 입력값을 바꿔보며 확인해 보세요.")

    with tab_pred:
        st.subheader("환자 정보 입력")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("나이", 20, 90, 55)
            sex = st.radio("성별", list(LABELS["sex"]), horizontal=True)
            cp = st.selectbox("가슴통증 유형 (cp)", list(LABELS["cp"]))
            trestbps = st.slider("안정 시 혈압 (trestbps)", 90, 200, 130)
            chol = st.slider("콜레스테롤 (chol)", 120, 570, 240)
        with col2:
            fbs = st.radio("공복 혈당 mg/dl (fbs)", list(LABELS["fbs"]), horizontal=True)
            restecg = st.selectbox("안정 시 심전도 (restecg)", list(LABELS["restecg"]))
            thalach = st.slider("최대 심박수 (thalach)", 70, 210, 150)
            exang = st.radio("운동 유발 협심증 (exang)", list(LABELS["exang"]), horizontal=True)
        with col3:
            oldpeak = st.slider("운동 후 ST 하강 (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
            slope = st.selectbox("ST 기울기 (slope)", list(LABELS["slope"]))
            ca = st.slider("주요 혈관 수 (ca)", 0, 4, 0)
            thal = st.slider("탈륨 검사 코드 (thal)", 0, 3, 2)

        row = pd.DataFrame([[age, LABELS["sex"][sex], LABELS["cp"][cp], trestbps, chol,
                             LABELS["fbs"][fbs], LABELS["restecg"][restecg], thalach,
                             LABELS["exang"][exang], oldpeak, LABELS["slope"][slope], ca, thal]],
                           columns=FEATURES)
        # [왜] predict()의 0/1보다 predict_proba()의 확률이 훨씬 많은 정보를 준다 —
        #      51%와 99%는 같은 "1"이지만 전혀 다른 상황이다(m3_penguins와 동일한 이유).
        proba = float(model.predict_proba(row)[0][1])
        st.divider()
        r1, r2 = st.columns([1, 2])
        r1.metric("심장병 위험 확률", f"{proba:.1%}")
        r2.progress(proba, text="모델이 추정한 위험도 (RandomForest, 학습 데이터 기준)")
        if proba >= 0.5:
            st.error("모델 판정: 위험군에 가깝습니다 — 입력값(cp·thalach·oldpeak)을 바꿔보며 어떤 요인이 민감한지 관찰해 보세요.")
        else:
            st.success("모델 판정: 정상군에 가깝습니다 — 입력값을 바꿔보며 경계가 어디서 뒤집히는지 관찰해 보세요.")
        st.caption("확률은 이 데이터로 학습한 모델의 추정일 뿐, 보정(calibration)되지 않은 값입니다.")


if __name__ == "__main__":
    main()
