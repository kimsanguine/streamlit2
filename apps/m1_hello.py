# 첫 Streamlit 앱 — 실행 모델(rerun)을 눈으로 확인하는 용도
# [권장] st.set_page_config는 import 직후 첫 st 명령으로 두는 것이 관례입니다(브라우저 탭 제목·레이아웃을 먼저 확정)
#        참고: 이 특강의 Streamlit 1.59.1에선 순서가 어긋나거나 여러 번 호출해도 에러 없이 additive하게 적용됩니다
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="첫 Streamlit 앱", page_icon="🎈", layout="centered")

# [핵심] 제목은 페이지에 보통 하나만 둡니다
st.title("🎈 나의 첫 Streamlit 앱")
# [핵심] 시계 트릭 — 위젯을 하나만 건드려도 이 시각이 바뀝니다. 바뀐다 = 스크립트 전체가 다시 실행됐다는 증거
st.caption(f"마지막 실행 시각: {time.strftime('%H:%M:%S')}")

# [widget] 위젯 3종 — text_input(문자열)·slider(범위)·selectbox(목록) 순서로 값을 받습니다
name = st.text_input("이름을 입력하세요", value="수강생")
level = st.slider("Streamlit 숙련도 (1~10)", 1, 10, 3)
interest = st.selectbox("오늘 가장 궁금한 주제", ["실행 모델(rerun)", "session_state", "캐싱", "배포"])

# [핵심] 위젯 값은 그 rerun 시점의 값을 즉시 돌려주는 변수일 뿐입니다(5절 참고)
st.write(f"안녕하세요, **{name}**님! 오늘 숙련도 {level}점으로 시작해서 '{interest}'를 배웁니다.")

# [흐름] 매 rerun마다 새 난수가 생성됩니다 — 슬라이더를 움직일 때마다 그래프 모양이 바뀌는 것으로도 rerun을 확인할 수 있습니다
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])

# [layout] columns — 화면을 가로로 n등분해 나란히 배치할 때 씁니다
col1, col2 = st.columns(2)
col1.metric("현재 숙련도", level, delta=level - 3)
col2.metric("목표 숙련도", 10, delta=10 - level)

# [layout] tabs — 같은 데이터를 화면 전환으로 보여줄 때 씁니다
tab_chart, tab_table = st.tabs(["📈 차트", "📋 통계 요약"])
with tab_chart:
    st.line_chart(chart_data)
with tab_table:
    st.dataframe(chart_data.describe())

# [layout] expander — 자주 안 쓰는 옵션을 접어둘 때 씁니다. checkbox — 켜기/끄기 토글
with st.expander("⚙️ 고급 설정 (평소엔 접어두고, 필요할 때만 펼침)"):
    show_raw = st.checkbox("원본 데이터 20행 보기")
    if show_raw:
        st.dataframe(chart_data)

# [흐름] 사이드바는 본문과 분리된 별도 영역 — 설정용 위젯을 모아두는 관용적 위치
st.sidebar.header("⚙️ 오늘의 설정")
st.sidebar.write(f"선택한 이름: {name}")
st.sidebar.write(f"관심 주제: {interest}")
# [widget] radio — 선택지가 2~4개로 적을 때 selectbox 대신 씁니다(항상 다 보여서 클릭 한 번에 선택)
study_mode = st.sidebar.radio("오늘 학습 방식", ["실습 위주", "이론 위주"])
st.sidebar.caption(f"선택: {study_mode}")
