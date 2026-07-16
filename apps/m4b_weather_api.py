# ST4b — 공개 API 연동 (requests): 날씨 API를 불러 Streamlit에 표시
# [왜] LLM(OpenAI)도 결국 HTTP API 호출이다. 여기서 '외부 API를 requests로 부르고 JSON을 받아 화면에
#      표시'하는 패턴을 손으로 익히면, ST5b에서 client.chat.completions.create가 같은 패턴임을 이해한다.
# [흐름] 17강 get_weather 도구(딕셔너리 mock)의 '진짜 API 버전'. 무키 공개 API(open-meteo)라 키 없이 동작.
# 실행: python3.11 -m streamlit run apps/m4b_weather_api.py

import requests
import streamlit as st

# [주의] 교실 네트워크가 막히면 USE_REAL_HTTP=False로 두면 mock 응답으로 흐름을 체감한다(17강 계승).
USE_REAL_HTTP = True

# open-meteo는 좌표(위도·경도)로 조회한다 — 무키(키 발급 불필요) 공개 API.
CITIES = {
    "서울": (37.57, 126.98), "부산": (35.18, 129.08), "제주": (33.51, 126.52),
    "대전": (36.35, 127.38), "광주": (35.16, 126.85),
}


# [왜] cache_data(ttl=600) — 성공 응답만 캐시한다. 예외는 캐시 밖에서 처리해, 일시적 네트워크 실패가
#      10분간 굳어버리는(복구돼도 계속 실패) 함정을 막는다(M2 cache_data + 에러처리 결합).
@st.cache_data(ttl=600)
def _fetch_weather_cached(lat: float, lon: float) -> dict:
    """open-meteo 호출 → JSON dict. 실패 시 예외를 던져(캐시 안 됨) 상위에서 처리하게 한다."""
    if not USE_REAL_HTTP:
        return {"current": {"temperature_2m": 21.0, "relative_humidity_2m": 60, "wind_speed_10m": 2.0}, "_mock": True}
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"}
    # [핵심] requests.get(url, params) → raise_for_status() → .json(). 실패하면 raise_for_status가 예외를 던진다.
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def fetch_weather(lat: float, lon: float) -> dict:
    """캐시된 호출을 감싸 예외만 {"_error"}로 변환한다 — 에러는 캐시되지 않아 재시도 시 다시 시도된다."""
    try:
        return _fetch_weather_cached(lat, lon)
    except requests.RequestException as e:  # 네트워크 계열만 잡아 안내(코드 버그성 예외는 삼키지 않게)
        return {"_error": str(e)}


def main():
    st.set_page_config(page_title="공개 API 연동 — 날씨", page_icon="🌤️", layout="centered")
    st.title("🌤️ 공개 API 연동 — 날씨 조회")
    st.caption("requests로 무키 공개 API(open-meteo)를 부르고 JSON을 받아 표시 — LLM API 호출과 같은 패턴")

    city = st.selectbox("도시 선택", list(CITIES))
    lat, lon = CITIES[city]

    if st.button("날씨 조회", type="primary"):
        with st.spinner("API 호출 중..."):
            data = fetch_weather(lat, lon)

        if "_error" in data:
            st.error(f"API 오류 — 네트워크를 확인하거나 코드의 USE_REAL_HTTP=False로 mock을 쓰세요: {data['_error'][:80]}")
        else:
            cur = data.get("current")
            if not cur:
                st.error("응답에 current 필드가 없습니다")
                return
            if data.get("_mock"):
                st.info("🧪 mock 응답(오프라인 모드) — 실제 API 대신 가짜 값입니다.")
            c1, c2, c3 = st.columns(3)
            c1.metric("기온", f"{cur.get('temperature_2m', '-')}°C")
            c2.metric("습도", f"{cur.get('relative_humidity_2m', '-')}%")
            c3.metric("풍속", f"{cur.get('wind_speed_10m', '-')} m/s")
            with st.expander("원본 JSON 보기 — LLM 응답도 이렇게 JSON으로 온다"):
                st.json(data)


if __name__ == "__main__":
    main()
