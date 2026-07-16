import streamlit as st

# [왜] `streamlit run showcase/Home.py`는 showcase/ 폴더만 import 경로에 넣는다 — `from showcase...`가
#      되려면 프로젝트 루트(이 파일의 상위 폴더)를 경로에 직접 추가해야 어떤 실행 방식·위치에서도 동작한다.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from showcase.core.theme import apply_theme, card, hero, kpi_row, selling_badge, theme_toggle

st.set_page_config(page_title="ML 쇼케이스", page_icon="🎯", layout="wide")
apply_theme()
theme_toggle()

hero(
    "키 없이, 지금 눌러보는 tool-calling 에이전트",
    "설치도 로그인도 API 키도 없이 — 이 화면에서 직접 만든 ML 모델 4개(펭귄·이미지·감성·심장병)와 내 문서 검색을 "
    "에이전트가 도구로 골라 호출합니다.",
)

# [내러티브] 핵심 셀링포인트를 KPI 한 칸이 아니라 전폭 배지로 승격 — 리크루터 시선 독점.
selling_badge(
    "<strong>API 키도 로그인도 없이</strong> — 링크를 연 그 순간 실제 도구 호출을 봅니다. "
    "키가 있으면 LLM이, 없으면 데모 모드가 <strong>같은 도구</strong>를 호출하고, "
    "<strong>왜 그 도구를 불렀는지 호출 trace가 그대로</strong> 표시됩니다.",
    icon="🔑",
)

# [구성] above-the-fold에 진입 동선 — 사이드바 탐색에 의존하지 않게 데모로 바로 점프.
st.markdown("#### 지금 바로 써보기")
d1, d2, d3 = st.columns(3)
with d1:
    with st.container(border=True):
        st.markdown("### 🐧 펭귄 분류")
        st.caption("RandomForest · 측정값 4개로 팔머펭귄 3종 분류")
        st.page_link("pages/1_🐧_펭귄_분류.py", label="펭귄 분류 열기 →", use_container_width=True)
with d2:
    with st.container(border=True):
        st.markdown("### 🖼️ 이미지 분류")
        st.caption("MobileNetV2 · ImageNet 1000종 top-5")
        st.page_link("pages/2_🖼️_이미지_분류.py", label="이미지 분류 열기 →", use_container_width=True)
with d3:
    with st.container(border=True):
        st.markdown("### 💬 감성 분석")
        st.caption("KoELECTRA · 한국어 문장 긍정/부정")
        st.page_link("pages/3_💬_감성_분석.py", label="감성 분석 열기 →", use_container_width=True)

d4, d5, _ = st.columns(3)
with d4:
    with st.container(border=True):
        st.markdown("### 🫀 심장병 예측")
        st.caption("RandomForest · UCI 1,025명 · 13개 지표 위험 확률")
        st.page_link("pages/5_🫀_심장병_예측.py", label="심장병 예측 열기 →", use_container_width=True)
with d5:
    with st.container(border=True):
        st.markdown("### 📚 RAG 챗봇")
        st.caption("내 문서 업로드 · 청킹 → 검색 → LLM 생성 답변")
        st.page_link("pages/6_📚_RAG_챗봇.py", label="RAG 챗봇 열기 →", use_container_width=True)

# 포트폴리오 하이라이트 — 에이전트 배너를 above-the-fold에 전폭 노출.
with st.container(border=True):
    c_txt, c_cta = st.columns([3, 1], vertical_alignment="center")
    with c_txt:
        st.markdown("### 🤖 tool-calling 에이전트")
        st.caption(
            "위 5개(펭귄·이미지·감성·심장병·문서 검색)를 도구로 지휘하는 에이전트 — 아무 문장이나 넣으면 "
            "어떤 도구를 왜 불렀는지 trace가 그대로 보입니다."
        )
    with c_cta:
        st.page_link("pages/4_🤖_에이전트.py", label="지금 에이전트 써보기 →", use_container_width=True)

# 증거 수치 — 실측만, 이모지·괄호 제거. 87ms는 이미지 도구 한정임을 라벨로 명시(과장 방지).
kpi_row(
    [
        {"label": "ML 모델", "value": "4종"},
        {"label": "이미지 추론 · CPU", "value": "~87ms"},
        {"label": "테스트 통과", "value": "21"},
        {"label": "Provider · 무료 포함", "value": "3"},
    ]
)

st.caption("이스트소프트 KDT AI Human 6기 강사·PM 출신이 직접 설계·구현했습니다.")

# 만드는 과정 — 진입 동선을 밀어내지 않게 접어둔다(관심 생긴 방문자만 펼침).
with st.expander("🧩 이 프로젝트는 어떻게 만들었나 — 문제·설계·구현·배포"):
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        card("문제", "에이전트는 API만으론 안 보인다 — 써보고 공유할 '얼굴'이 필요하다.", icon="❓")
    with e2:
        card("설계", "교안 앱 로직을 재사용하는 단방향 core 레이어링(models→tools→agent).", icon="🧩")
    with e3:
        card("구현", "OpenAI 호환 tool-calling 루프 + 키 없이 동작하는 fake 폴백.", icon="⚙️")
    with e4:
        card("배포", "Streamlit Cloud(CPU torch) + Secrets로 키 분리.", icon="🚀")

    st.markdown("**기술 스택**")
    st.markdown(
        "`Streamlit` · `scikit-learn(RandomForest)` · `PyTorch(MobileNetV2)` · "
        "`Transformers(KoELECTRA)` · `OpenAI 호환 provider(local/openrouter/openai)`"
    )
    st.markdown("**아키텍처** — 자세히는 `docs/ARCHITECTURE.md` 참고")
    st.markdown(
        "```\n"
        "사용자 → 에이전트(provider 루프) → 5 tool → core.models → 교안 앱(apps/*)\n"
        "                                   ├─ classify_penguin  (RandomForest)\n"
        "                                   ├─ analyze_sentiment (KoELECTRA)\n"
        "                                   ├─ classify_image    (MobileNetV2)\n"
        "                                   ├─ predict_heart     (RandomForest)\n"
        "                                   └─ search_docs       (TF-IDF/임베딩 검색)\n"
        "```"
    )
