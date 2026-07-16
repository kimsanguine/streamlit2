# cache_data로 느린 함수를 캐싱 전후 비교 — time.sleep(3)으로 "무거운 작업"을 흉내냄
import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="캐싱 실습", page_icon="⏳")
st.title("⏳ 캐싱 전후 체감")

# [왜] time.sleep(3)으로 "무거운 데이터 로드"를 흉내 — 실제로는 대용량 CSV 읽기, 모델 로드 등에 해당
def load_data_slow():
    time.sleep(3)
    return pd.DataFrame({"a": range(5), "b": range(5, 10)})

@st.cache_data
def load_data_cached():
    time.sleep(3)
    return pd.DataFrame({"a": range(5), "b": range(5, 10)})

st.subheader("캐싱 없이 (매 클릭마다 3초)")
if st.button("느린 로드 실행", key="btn_slow"):
    start = time.time()
    df = load_data_slow()
    st.write(f"소요 시간: {time.time() - start:.2f}초")
    st.dataframe(df)

st.divider()

st.subheader("cache_data 적용 (첫 클릭만 3초, 이후 즉시)")
if st.button("캐싱된 로드 실행", key="btn_cached"):
    start = time.time()
    df = load_data_cached()
    st.write(f"소요 시간: {time.time() - start:.2f}초")
    st.dataframe(df)

st.caption("버튼을 두 번 이상 눌러 비교하세요 — 캐싱 버전은 두 번째 클릭부터 3초 대기가 사라집니다.")