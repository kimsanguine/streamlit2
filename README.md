# 🎈 Streamlit 프로젝트 특강 — 하루 완성 (ST1~ST5)

> ✅ 2026-07-15 확정 — **정식 교안은 [`강의_AI_Pair/`](강의_AI_Pair/)의 ST1~ST5 통합본**입니다(검증팀 5렌즈 99.1/100 — 2026-07-13 8→5파일 통합본 채점 기준. ST1 재구성·ST3 재설계·ST4 RAG 신규·ST5 showcase 배포 등 2026-07-15 v2 변경분은 이번 수정 라운드에서 개별 지적 대응으로 반영됨, 별도 재채점 전).

> 20강 통합 미니 프로젝트에서 만든 `agent_app/app.py`의 "얼굴"을 제대로 이해하고 업그레이드하는 하루입니다.
> AI Python 3(Part I. Python) 20강 커리큘럼의 직속 후속 특강이며, AI Human 6기 커리큘럼 기준으로는
> Phase 2(NLP·Voice·Vision) 실습 모델을 웹 데모로 보여주는 공용 도구이자, Phase 3(Agent) 실습에서
> 에이전트 UI(ST4 채팅 패턴)로 다시 쓰입니다.

---

## 🎯 쇼케이스 — tool-calling 에이전트 (포트폴리오 레이어)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![tests](https://img.shields.io/badge/tests-17%20passed-2EA043)
![키 없이 동작](https://img.shields.io/badge/데모%20모드-키%20없이%20동작-7C3AED)

**교안(ST1~ST5)에서 만든 3개 ML 모델을 하나의 tool-calling 에이전트가 지휘하는 포트폴리오** — API 키가 없어도 데모 모드로 실제 도구를 호출한다.

이 저장소는 **이중 레이어**다. `apps/`(교안 — 비전공자가 셀 단위로 따라 만드는 학습 자료) 위에, 그 앱 로직을 **재사용**하는 `showcase/`(모던 SaaS 디자인의 에이전트)가 올라간다. 의존 방향은 단방향(`showcase` → `apps`)이라 교안은 독립적으로 계속 쓰인다.

<!-- 히어로 스크린샷 자리 (Phase D에서 촬영해 삽입): ![쇼케이스 홈](docs/hero.png) -->

| 데모 | 모델 | 하이라이트 |
|---|---|---|
| 🐧 펭귄 분류 | RandomForest | Train/Test 병기·CV·permutation importance·미보정 캡션 |
| 🖼️ 이미지 분류 | MobileNetV2 (ImageNet 1000종) | CPU 추론 ~87ms·top-5 |
| 💬 감성 분석 | KoELECTRA-small (NSMC) | 한국어 긍/부정·`truncation` 견고성 |
| 🤖 **에이전트** | OpenAI 호환 provider 루프 | 위 3개를 **tool**로 호출·**키 없이 데모 모드**·tool trace 전면 노출 |

```bash
# 쇼케이스 실행 (프로젝트 루트에서)
python3.11 -m streamlit run showcase/Home.py
```

- **provider 3종**(20강 규약 계승): `local`(Ollama) · `openrouter`(무료) · `openai`. `base_url`만 전환하며 **Anthropic/Claude는 쓰지 않는다**. `[개발자 트랙]`.
- **배포**: Streamlit Community Cloud(무료 티어). torch는 CPU 인덱스(`requirements.txt` 최상단), 한글 폰트는 `packages.txt`(fonts-nanum), 키는 `.streamlit/secrets.toml`(gitignore)로 분리.

> 아래는 이 쇼케이스를 **만드는 과정**을 하루 특강으로 담은 교안(ST1~ST5, 강의_AI_Pair/)이다.

---

## 🚀 시작하기

> ⏱️ **"3분 안에"는 전날 `수업전_환경체크.md`로 설치를 끝낸 경우입니다.** 처음 설치라면 torch 등으로 5~15분 걸립니다.

> 🪟 **Windows**: 아래·이후 모든 `python3.11`은 **`py -3.11`**로 바꿔 실행하세요(자세한 규칙은 `수업전_환경체크.md`).

터미널이 처음이라면 **`수업전_환경체크.md`의 "0. 터미널 여는 법"**을 먼저 보세요(터미널 열기·폴더 드래그로 경로 자동 입력).

```bash
python3.11 -m pip install -r requirements.txt   # 첫 설치는 torch 포함 5~15분 — 멈춘 듯해도 끄지 마세요
python3.11 -m streamlit run apps/m1_hello.py
```

- 실행하면 브라우저가 자동으로 열립니다. **안 열리면** 터미널에 뜬 `Local URL`(보통 `http://localhost:8501`)을 복사해 주소창에 붙여넣으세요.
- 앱을 끝낼 때는 터미널에서 **`Ctrl + C`**.
- `pip`/`streamlit`을 그냥 쓰지 않고 `python3.11 -m`을 붙이는 이유는, 파이썬이 여러 개인 PC에서 엉뚱한 파이썬이 실행되는 것을 막기 위해서입니다(아래 FAQ 참고).

## 🎯 학습 목표
- Streamlit 핵심 문법(rerun·위젯·`session_state`·캐싱·layout)을 한 번에 체득한다 (ST1·ST2)
- 시각화·EDA를 인터랙티브 웹 대시보드로 만든다 — 내 미니프로젝트 CSV로도 (ST2)
- 딥러닝 모델(이미지·감성)을 웹 데모로 시각화한다 (ST3)
- 챗 UI 4요소 + 비스트리밍 tool-calling 루프로 "도구 쓰는 에이전트"를 직접 만든다 (ST4)
- git CLI 기본 명령(add/commit/push)과 Streamlit Cloud로 내 앱에 URL을 달고, 20강 앱을 업그레이드한다 (ST5)

## 📂 노트북 구성 — [`강의_AI_Pair/`](강의_AI_Pair/)

| 노트북 | 내용 | 주요 산출물 |
|---|---|---|
| `ST1_Streamlit핵심문법(AI_Pair).ipynb` | 문법 코어(rerun·위젯 카탈로그·레이아웃) | `apps/ex1_write.py`~`ex5_clock.py`·`m1_hello.py` 등 |
| `ST2_시각화EDA대시보드(AI_Pair).ipynb` | 펭귄 인터랙티브 EDA·분류 시각화 | `apps/m3_penguins.py` (+`eda_template.py`) |
| `ST3_딥러닝데모앱(AI_Pair).ipynb` | 이미지 분류 + 한국어 감성분석 | `apps/m4_image.py`·`m4_sentiment.py` |
| `ST4_에이전트앱(AI_Pair).ipynb` | 챗 UI + 비스트리밍 tool-calling Lab | `apps/m5_chatbot.py`·`m5b_agent_loop.py` (+`rag_lite.py`) |
| `ST5_배포미니프로젝트(AI_Pair).ipynb` | git CLI·Cloud 배포·20강 앱 업그레이드 | 배포 URL |

각 노트북 끝의 **🚀 배포 빌드업 [N/5]** 가 ST1(저장소 생성·첫 push)부터 ST5(최종 배포)까지 하루 내내 누적됩니다 — 상세는 [`deploy/배포_튜토리얼.md`](deploy/배포_튜토리얼.md).

## 🗓️ 하루 시간표 (필수 경로 345분 + 버퍼)

| 시간 | 노트북 | 주제 | 분 |
|---|---|---|---|
| 09:30–09:45 | — | 오리엔테이션 (환경·GitHub 계정은 사전 과제) | 15 |
| 09:45–10:45 | ST1 | Streamlit 핵심 문법 + 첫 push | 60 |
| 10:55–12:05 | ST2 | 시각화·EDA 대시보드 | 70 |
| 12:05–13:05 | 점심 | | |
| 13:05–14:05 | ST3 | 딥러닝 데모 앱 | 60 |
| 14:15–15:30 | ST4 | 에이전트 앱 (챗 UI + tool-calling Lab) | 75 |
| 15:40–17:00 | ST5 | 배포 + 미니 프로젝트 · URL 공유 | 80 |
| 17:00–17:10 | — | 마무리 | 10 |

## ⏰ 시간 조정 가이드
- 🧭 **모듈 사이 10분 휴식이 buffer입니다.** 지연은 가장 뒤(ST5 배포)로 몰리므로, 밀리면 아래 순서로 생략하세요:
  ① 각 노트북의 (선택) 구간 → ② 심화(`<details>`) 전부 → ③ ST3 감성분석을 강사 데모로(교안 기본값).
- **ST5가 쫓기면**: 배포까지 못 가도 됩니다 — **로컬 동작 스크린샷 + 업그레이드한 코드** 제출로 완료 인정.
  시간에 쫓겨 "폴더 통째 업로드"로 `.env`(키)를 노출하는 사고를 막는 저비용 탈출구입니다.
- **빠르면**: 각 노트북의 "📌 오늘의 도전 과제(선택)" — 내 미니프로젝트 대시보드(ST2)·내 문서 RAG 에이전트(ST4) — 를 수업 중 시작합니다.

## 🧪 실행 환경

검증 기준: 2026-07-13, `python3.11` — 정확한 버전 고정과 사전 설치 절차는 이 폴더의
**`수업전_환경체크.md`**와 **`prerequisite.py`**를 따르세요(ST3 모델 사전 캐싱, seaborn 데이터셋 캐싱 등
수업 전날 끝내둘 항목이 정리돼 있습니다).

```txt
streamlit>=1.58   # 실측 1.59.1
pandas            # 실측 2.3.3
numpy             # 실측 2.4.4
scikit-learn      # 실측 1.9.0 — ST2 펭귄 분류기(RandomForest)
seaborn           # 실측 0.13.2 — ST2 팔머펭귄 데이터셋 로드 + 시각화
matplotlib        # 실측 3.10.8 — seaborn 백엔드
torch             # 실측 2.13.0 — ST3 MobileNetV2 추론
torchvision       # 실측 0.28.0 — ST3 MobileNetV2 가중치·전처리
transformers      # 실측 5.6.2 — ST3 KoELECTRA 감성분석 pipeline
pillow            # 실측 12.2.0 — ST3 이미지 파일 열기(PIL.Image)
```

(전체 버전 고정 목록은 이 폴더의 `requirements.txt` 원본을 그대로 사용하세요 — 위 표는 요약입니다.)

## 🛠️ 트러블슈팅 FAQ

| 증상 | 원인 | 해결 |
|---|---|---|
| (옛 자료의) `set_page_config must be first` 에러 | 이 특강의 Streamlit 1.59.1에선 **발생하지 않음**(additive) | 순서와 무관하게 동작하나, 가독성상 맨 위 두는 것 권장 (ST1 참고) |
| `StreamlitDuplicateElementId` (옛 이름 `DuplicateWidgetID`) | 반복문 안 위젯에 고유 `key`가 없음 | `key=f"w_{i}"`처럼 반복마다 고유 key 지정 |
| 버튼을 눌렀는데 위젯 결과가 사라짐/초기화됨 | 반환값 기반 처리(안티패턴) | 콜백(`on_click`) + `session_state` 플래그로 전환 |
| 위젯을 두 번 눌러야 반영됨 | `session_state` 수정이 위젯 생성 **이후**에 일어남 | state 수정은 반드시 위젯 생성 **전** 코드 흐름에서 실행 |
| 클라우드에서 리소스 초과(OOM) | 모델이 rerun마다 재로드됨 | `@st.cache_resource` 적용, 그래도 부족하면 `ttl`·`max_entries` 조정 또는 더 작은 모델로 축소 |
| `cache_resource` 객체를 수정했더니 다른 세션에도 반영됨 | 정상 동작 — 싱글턴을 공유하기 때문 | 세션별 독립 변형이 필요하면 `cache_data`를 쓰거나 반환값을 세션 안에서 복사해서 사용 |
| `streamlit` 명령이 `ModuleNotFoundError` 등을 냄 | 파이썬이 여러 개 설치된 PC에서 다른 파이썬의 streamlit이 실행됨 | `python3.11 -m streamlit run apps/....py`로 실행 파이썬을 고정 |
| 노트북 셀에서 `streamlit run`을 실행했더니 셀이 안 끝남 | 서버 프로세스가 셀을 점유한 채 계속 떠 있음 | 노트북 안에서는 절대 실행하지 않고 반드시 **터미널**에서 실행 |

## 🧪 수정해서 시도해볼 것
1. **ST1** `apps/m1_hello.py` — 사이드바 radio로 line/bar/area 차트를 실시간 전환해보세요.
2. **ST2** `apps/ex3_cache.py` — `RandomForestClassifier(n_estimators=50, ...)`을 `n_estimators=500`으로 늘리고, 버튼을 두 번 눌러 첫 번째(캐시 미스)와 두 번째(캐시 히트) 소요 시간 차이가 얼마나 벌어지는지 확인해보세요.
3. **ST4** `apps/m5_chatbot.py` — `fake_llm_stream`의 단어 간격(`0.05`)을 `0.2`로 늘려 "너무 느린 스트리밍"이 사용자 경험에 어떤 영향을 주는지 관찰해보세요.
4. **ST5 미니프로젝트** — 20강 앱에 필수 3종(`session_state`·`cache_resource`·`st.status`)을 적용한 뒤, 선택 확장 1종(이미지 분류 tool·CSV 일괄 감성분석·통계 대시보드·멀티모달 첨부) 중 하나를 추가해보세요.

## 🔗 Phase 연결
이 특강은 AI Python 3(Part I. Python) 20강 통합 프로젝트의 직속 후속입니다. AI Human 6기 커리큘럼 기준으로는
Phase 2(NLP·Voice·Vision) 실습에서 만들 모델들을 곧바로 웹 데모로 보여주는 공용 도구이며, Phase 3(Agent)
실습에서 에이전트 UI(ST4 패턴)로 다시 쓰입니다.
