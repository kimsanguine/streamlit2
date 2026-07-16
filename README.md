# Streamlit 특강 — 데모 앱 & 배포 저장소

AI Human 과정 Streamlit 특강의 **실행 가능한 데모 앱 모음**입니다. 수업 교안(노트북)은 이 저장소에 포함되지 않고 수업 자료로 별도 제공됩니다 — 이 저장소는 앱 실행과 Streamlit Cloud 배포에 필요한 것만 담습니다.

## 빠른 시작

```bash
pip install -r requirements.txt
python -m streamlit run apps/m3_penguins.py
```

> Windows에서 python 명령이 여럿이라면 `py -3.11 -m streamlit run ...` 형태를 사용하세요. 환경 점검은 `수업전_환경체크.md`, 모델 사전 다운로드는 `python prerequisite.py`.

## 앱 구성 (`apps/`)

| 앱 | 내용 | 키 필요 |
|---|---|---|
| `m1_hello.py` | Streamlit 기본 문법 데모 | ✗ |
| `m3_penguins.py` | 펭귄 분류 인터랙티브 EDA + ML (배포 1순위 — 가장 가벼움) | ✗ |
| `heart_disease.py` | **심장병 위험 예측 대시보드** — EDA + 13피처 입력 예측 (`data/heart.csv` 동봉, UCI CC BY) | ✗ |
| `eda_template.py` | CSV 업로드 EDA 템플릿 (한국어 Excel CSV 지원) | ✗ |
| `m4_image.py` | 이미지 분류 + occlusion 히트맵 (torch) | ✗ |
| `m4_sentiment.py` | 한국어 감성분석 일괄 채점 데모 (KoELECTRA) | ✗ |
| `m4b_weather_api.py` | 공개 날씨 API 대시보드 | ✗ |
| `m5_chatbot.py` · `m5b_agent_loop.py` | 챗 UI · tool-calling 에이전트 루프 | 없으면 데모 모드 |
| `rag_lite.py` | 문서 업로드 → 청킹 → TF-IDF 검색 (무키 RAG) | ✗ |
| `rag_chatbot.py` | **PDF RAG 챗봇** — 업로드 → 청킹 → 임베딩 검색 → LLM 생성 답변 | 없으면 검색 데모 모드 |

`ex*.py`는 수업 실습에서 각자 완성하는 스켈레톤이라 저장소에 포함하지 않습니다.

## API 키 설정 (선택 — LLM·임베딩 기능)

저장소 루트에 `.env` 파일을 만들면 자동 로드됩니다(코드에 키를 적지 마세요):

```
OPENROUTER_API_KEY=발급받은_키
OPENAI_API_KEY=발급받은_키
```

- 로컬 무료 경로: [Ollama](https://ollama.com) 설치 후 `ollama pull hermes3:8b`, `ollama pull nomic-embed-text`
- 키/서버가 하나도 없어도 앱은 죽지 않고 데모(검색) 모드로 동작합니다. 앱이 가용한 provider를 자동 감지하고, 호출 실패 시 다른 provider로 폴백합니다.
- Streamlit Cloud에서는 앱 설정의 **Secrets**에 같은 키를 넣으면 됩니다.

## Streamlit Cloud 배포

절차는 [`deploy/배포_튜토리얼.md`](deploy/배포_튜토리얼.md)를 따르세요. 요약:

1. [share.streamlit.io](https://share.streamlit.io) → GitHub 로그인 → **Create app**
2. 이 저장소 / `main` / **Main file path: `apps/m3_penguins.py`**
3. Deploy — 루트 `requirements.txt`(torch CPU 인덱스 포함)와 `packages.txt`(한글 폰트)가 자동 적용됩니다

## 포트폴리오 쇼케이스 (`showcase/`)

교안 앱들의 모델을 **tool-calling 에이전트의 도구로 재사용**하는 포트폴리오급 멀티페이지 앱입니다(디자인 시스템·다크 토글 포함):

```bash
python -m streamlit run showcase/Home.py   # 저장소 루트에서 실행
```
