# ST3 — 딥러닝 데모 ②: 한국어 감성분석 (KoELECTRA) — 채점 모드 + 단어 기여도
# [왜] 14강에서 pipeline("sentiment-analysis", model="sangrimlee/bert-base-multilingual-cased-nsmc")를
#      이미 썼다. 같은 패턴, 모델만 더 작은 것(56.5MB, 실측 — 강의실 네트워크 배려)으로 바꾸고
#      여기에 웹 UI를 입힌다. 오늘은 "정답이 있는 시험"으로 채점까지 해서 모델의 진짜 실력을 잰다.
# 실행: python3.11 -m streamlit run apps/m4_sentiment.py

import html
import logging

import pandas as pd
import streamlit as st

# [왜] 팔레트 상수 — 강사 교안 배색(TEAL=긍정 #0E6B56, CORAL=부정 #993C1D)의 RGB값.
#      표 배경색·단어 하이라이트에서 rgba(r,g,b,투명도) 형태로 재사용하려고 튜플로 둔다.
TEAL_RGB = (14, 107, 86)
CORAL_RGB = (153, 60, 29)

# [실측] 내장 라벨 20문장 벤치마크 — "정답을 미리 아는 시험"으로 모델의 진짜 정답률을 잰다.
#        영화 리뷰 14개(감성분류 fine-tuning 도메인, NSMC) + 쇼핑몰 리뷰 4개(도메인 밖) + 애매한 문장 2개.
#        (아래 정답률·오분류는 2026-07-15 실측 기준 — 문장을 바꾸면 수치도 달라질 수 있다)
BENCH_SENTENCES = [
    # 영화 리뷰 — 긍정 7문장(모델의 학습 도메인)
    ("이 영화 정말 감동적이었어요, 눈물이 났습니다", "긍정", "영화"),
    ("배우들의 연기가 훌륭하고 스토리도 탄탄했어요", "긍정", "영화"),
    ("올해 본 영화 중 최고예요, 강력 추천합니다", "긍정", "영화"),
    ("연출이 세련되고 음악도 좋았습니다", "긍정", "영화"),
    ("몰입감이 대단해서 시간 가는 줄 몰랐어요", "긍정", "영화"),
    ("이 감독의 전작보다 훨씬 좋아진 것 같아요", "긍정", "영화"),
    ("결말이 완벽해서 정말 만족스러웠습니다", "긍정", "영화"),
    # 영화 리뷰 — 부정 7문장
    ("완전 실망이었어요, 시간 낭비했습니다", "부정", "영화"),
    ("스토리가 너무 뻔하고 지루했어요", "부정", "영화"),
    ("연기가 어색하고 몰입이 안 됐습니다", "부정", "영화"),
    ("돈이 아까운 영화였어요", "부정", "영화"),
    ("편집이 산만해서 이해하기 어려웠어요", "부정", "영화"),
    ("기대했던 것보다 훨씬 별로였습니다", "부정", "영화"),
    ("다시는 보고 싶지 않은 영화예요", "부정", "영화"),
    # [왜] 도메인이탈 4문장 — 감성분류 fine-tuning은 "영화 리뷰"(NSMC)로만 했다(기반 모델의
    #      pretraining은 대규모 한국어). 쇼핑몰 리뷰는 fine-tuning 때는 본 적 없는 도메인이다 —
    #      여기서 무슨 일이 벌어지는지가 오늘의 핵심 질문이다.
    ("포장이 뜯어진 채로 왔어요", "부정", "도메인이탈"),
    ("재구매 의사 있습니다", "긍정", "도메인이탈"),
    ("사이즈가 작아 교환했어요", "부정", "도메인이탈"),
    ("고객센터 연결이 안 돼요", "부정", "도메인이탈"),
    # [왜] 애매 2문장 — 사람도 "긍정도 부정도 아닌" 애매함을 느끼는 표현. 모델도 흔들리는지 본다.
    ("그럭저럭 볼만은 했어요", "긍정", "애매"),
    ("나쁘진 않은데 다시 보고 싶진 않네요", "부정", "애매"),
]


# [왜] cache_resource — 56.5MB 모델을 매 rerun마다 다시 로드하면 안 된다. 첫 로드만 느리고
#      (실측 로드+추론 14.7s) 그다음부터는 캐시된 파이프라인을 즉시 재사용한다.
@st.cache_resource
def load_pipeline():
    # [왜] transformers는 import만 5초+ — 파일 최상단에 두면 '페이지가 뜨기'부터 느려진다.
    #      실제 모델을 쓰는 이 함수 안에서 import하면 앱은 즉시 뜨고, 로드는 첫 분석 때 1회만(스피너와 함께).
    from transformers import pipeline
    # [흐름] 모델 이름 한 줄만 바꾸면 다른 감성분석 모델로 즉시 교체할 수 있다
    return pipeline("sentiment-analysis", model="daekeun-ml/koelectra-small-v3-nsmc")


# [통계] analyze는 st 부작용 없는 순수 함수 — text를 인자로 받아 테스트에서 바로 호출 가능.
# [왜] truncation=True, max_length=512 — 긴 입력을 토큰 512개로 잘라 KoELECTRA의 입력
#      한도를 넘겨도 죽지 않게 한다(긴 문단을 붙여넣으면 KoELECTRA 입력 한도(512토큰)를 넘겨 앱이 죽던 것을 막는다).
def analyze(text: str) -> dict:
    # [흐름] 캐시된 파이프라인 재사용(첫 호출 이후엔 즉시 반환)
    clf = load_pipeline()
    # [주의] truncation·max_length로 512토큰을 넘는 입력도 죽지 않고 안전하게 자른다
    result = clf(text, truncation=True, max_length=512)[0]

    # [주의] ⚠️ 모델마다 라벨 형식이 다르다 — 이 모델(daekeun-ml/koelectra-small-v3-nsmc)은
    #        'positive'/'negative' 문자열이 아니라 '1'(긍정)/'0'(부정)을 반환한다.
    #        (실측: 긍정문 {'label': '1', 'score': 0.997}, 부정문 {'label': '0', 'score': 0.9995})
    #        → 교훈: 새 모델을 쓸 때는 항상 print(results)로 원본 출력을 먼저 눈으로
    #          확인한 뒤에 라벨 매핑 코드를 짜야 한다. "보통 이렇겠지"로 짐작하지 않는다.
    label = "긍정" if result["label"] == "1" else "부정"
    return {"label": label, "score": result["score"]}


# [캐시] cache_data — 같은 20문장·같은 모델이면 채점 결과도 항상 같다("데이터"이므로 cache_data).
#        캐시가 없으면 채점 모드 탭을 열 때마다(=rerun마다) 20번 추론을 다시 돌리게 된다.
@st.cache_data(show_spinner=False)
def run_benchmark() -> pd.DataFrame:
    rows = []
    # [흐름] 20문장을 하나씩 analyze() — "채점"이므로 정답(true_label)과 예측을 나란히 기록한다
    for text, true_label, category in BENCH_SENTENCES:
        out = analyze(text)
        rows.append(
            {
                "문장": text,
                "카테고리": category,
                "정답": true_label,
                "예측": out["label"],
                "확신도": out["score"],
                # [왜] 정답==예측이면 ✓, 아니면 ✗ — 표에서 오분류를 한눈에 찾기 위한 열
                "결과": "✓" if out["label"] == true_label else "✗",
            }
        )
    return pd.DataFrame(rows)


# [왜] occlusion(이미지 히트맵)과 정확히 같은 원리 — "가려보기"가 픽셀 칸을 가렸다면, 이번엔
#      단어를 하나씩 빼보고 원래 판정의 점수가 얼마나 흔들리는지 측정한다("빼보기", leave-one-out).
def leave_one_out(text: str):
    # 원문 전체의 판정 — 이 판정이 "얼마나 흔들리는지"를 기준으로 잰다
    base = analyze(text)
    # [주의] 형태소 분석이 아니라 공백 기준 — 교육용 근사(정교한 토큰화는 심화)
    words = text.split()
    contributions = []
    for i in range(len(words)):
        # [흐름] i번째 단어만 뺀 문장을 새로 조립 — 나머지 단어 순서는 그대로 유지
        variant = " ".join(words[:i] + words[i + 1:])
        if not variant.strip():
            continue
        out = analyze(variant)
        # [왜] pipeline은 "예측한 라벨"의 점수만 준다 — base["label"] 관점으로 점수를 다시 맞춰야
        #      "이 단어를 빼서 원래 판정이 얼마나 흔들렸는지"를 공정하게 비교할 수 있다.
        same_label_score = out["score"] if out["label"] == base["label"] else (1 - out["score"])
        # [흐름] 기여도 = 원래 점수 - 이 단어를 뺐을 때 점수 — 클수록 "이 단어가 판정을 떠받쳤다"는 뜻
        contributions.append((words[i], base["score"] - same_label_score))
    return base["label"], base["score"], contributions


# [왜] 기여도가 큰 단어일수록 라벨 색(긍정=TEAL/부정=CORAL)을 진하게 — 히트맵의 "뜨거운 색"과
#      같은 발상이다. 기여도가 0 이하(제거해도 판정이 안 흔들린 단어)는 배경색을 거의 안 준다.
def render_word_highlight(label: str, contributions: list[tuple[str, float]]) -> None:
    # 판정 라벨에 맞는 색을 고른다
    r, g, b = TEAL_RGB if label == "긍정" else CORAL_RGB
    # 정규화 기준 — 가장 큰 기여도를 100%로
    max_c = max((c for _, c in contributions), default=0.0)
    spans = []
    for word, c in contributions:
        # 음수 기여도는 0으로 잘라 하이라이트 없음
        intensity = max(c, 0) / max_c if max_c > 0 else 0.0
        # 0~1 범위의 배경 투명도 — 최소 0.15는 항상 보이게
        alpha = 0.15 + 0.7 * intensity
        # [주의] html.escape로 사용자가 입력한 단어를 이스케이프 — HTML 태그·따옴표가 그대로 삽입되는 것을 막는다
        spans.append(
            f'<span style="background-color:rgba({r},{g},{b},{alpha:.2f}); padding:3px 6px; '
            f'border-radius:4px; margin:2px; display:inline-block;">'
            f"{html.escape(word)}</span>"
        )
    st.markdown(" ".join(spans), unsafe_allow_html=True)  # unsafe_allow_html — 위에서 이미 이스케이프했으므로 안전


def main():
    # [왜] set_page_config는 다른 st.* 호출보다 먼저, 스크립트당 한 번만 와야 한다(공식 규칙)
    st.set_page_config(page_title="한국어 감성분석 데모", page_icon="💬", layout="centered")
    st.title("💬 한국어 감성분석 데모")
    st.caption('14강 pipeline("sentiment-analysis")의 웹 버전 — KoELECTRA-small (NSMC 파인튜닝)')

    # [흐름] 탭 순서 = 학습 순서 — 먼저 "정답이 있는 시험"으로 실력을 확인하고, 그다음 자유 탐색
    tab_bench, tab_free = st.tabs(["📝 채점 모드 (내장 20문장)", "✍️ 내 문장 분석 (단어 기여도)"])

    # [왜] tabs로 화면을 나눈다 — "정답이 있는 시험"(채점 모드)과 "자유 탐색"(내 문장 분석)은
    #      목적이 달라 한 화면에 욱여넣으면 오히려 헷갈린다(ST3 토론 절의 st.tabs 활용과 동일).
    with tab_bench:
        # [흐름] 채점 모드를 여는 순간 20문장 전체가 자동으로 채점된다 — 버튼을 따로 누를 필요 없음
        st.markdown(
            "모델이 한 번도 정답을 알려주지 않은 **20문장**(영화 리뷰 14 + 쇼핑몰 리뷰 4 + "
            "애매한 문장 2)을 일괄 채점합니다. 영화 리뷰는 감성분류 fine-tuning 도메인(NSMC)과 같고, "
            "쇼핑몰 리뷰는 **fine-tuning 때는 본 적 없는 도메인**입니다."
        )
        # cache_data 덕분에 재실행해도 20번 추론이 다시 돌지 않는다
        bench_df = run_benchmark()
        # [흐름] ✓ 개수의 비율 = 모델 정답률 — True/False 평균은 곧 True 비율
        accuracy = (bench_df["결과"] == "✓").mean()
        # [왜] 다수결 베이스라인 — "무조건 다수 라벨만 찍었을 때" 정답률. 모델 정답률과 비교해야
        #      모델이 실제로 뭔가 "배운" 건지, 그냥 다수쪽만 찍어도 되는 수준인지 판단할 수 있다.
        majority_label = bench_df["정답"].value_counts().idxmax()
        majority_baseline = (bench_df["정답"] == majority_label).mean()

        # 정답률·베이스라인·차이를 나란히 — ST2 지표 카드와 같은 패턴
        col1, col2, col3 = st.columns(3)
        col1.metric("모델 정답률", f"{accuracy * 100:.0f}%")
        col2.metric(f"다수결 베이스라인('{majority_label}' 찍기)", f"{majority_baseline * 100:.0f}%")
        # [왜] 차이가 클수록 "모델이 실제로 뭔가 배웠다"는 근거가 강해진다
        col3.metric("차이", f"{(accuracy - majority_baseline) * 100:+.0f}%p")

        # [왜] 결과 열(✓/✗) 기준으로 행 배경색을 입혀 오분류가 표에서 한눈에 도드라지게 한다.
        def highlight_row(row):
            # [흐름] ✓행은 TEAL(연하게), ✗행은 CORAL(진하게) — 오분류가 더 눈에 띄도록 알파값을 높인다
            rgb = TEAL_RGB if row["결과"] == "✓" else CORAL_RGB
            alpha = 0.15 if row["결과"] == "✓" else 0.25
            return [f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"] * len(row)

        # [흐름] Styler로 배경색을 입히고, format으로 확신도를 0.997 대신 99.7%로 보여준다
        st.dataframe(
            bench_df.style.apply(highlight_row, axis=1).format({"확신도": "{:.1%}"}),
            width="stretch",
            hide_index=True,
        )

        # [인라인 시각화] 카테고리별 정답률 — "어디서 틀리는지"를 막대그래프로 한눈에 보여준다.
        # [흐름] groupby("카테고리")로 영화/도메인이탈/애매를 나누고, 각각의 ✓ 비율을 구한다
        cat_acc = (bench_df.groupby("카테고리")["결과"].apply(lambda s: (s == "✓").mean()) * 100).rename("정답률(%)")
        st.bar_chart(cat_acc)  # 영화(학습 도메인)는 높고, 도메인이탈은 낮게 나오는지 눈으로 확인

        # [왜] 오분류만 따로 뽑아 확신도를 확인 — "틀렸을 때 얼마나 자신 있었는지"가 이번 교훈의 핵심
        wrong = bench_df[bench_df["결과"] == "✗"]
        if len(wrong) > 0:
            high_conf_wrong = wrong[wrong["확신도"] >= 0.9]
            st.caption(
                f"오분류 {len(wrong)}건 중 {len(high_conf_wrong)}건은 확신도 90%+ 로 "
                "**자신 있게** 틀렸습니다 — '배운 세계(영화 리뷰) 밖에서는 자신 있게 틀린다'는 "
                "교훈이 바로 이 지점입니다. 확신도 열을 보면 오분류가 항상 낮은 확신도로 "
                "일어나는 건 아니라는 것도 확인할 수 있습니다."
            )

    with tab_free:
        st.markdown(
            "문장을 하나 입력하면, 단어를 하나씩 빼보고 원래 판정이 얼마나 흔들리는지 재서 "
            "**어떤 단어가 판정에 가장 크게 기여했는지** 색으로 보여줍니다 — 이미지 탭(가려보기)과 "
            "정확히 같은 원리를 문장에 적용한 것입니다('빼보기', leave-one-out)."
        )
        # [왜] 기본값을 도메인 밖 오분류 문장으로 — 처음 열자마자 "왜 틀렸는지"를 바로 체감하게 한다
        text = st.text_input("문장 입력", value="포장이 뜯어진 채로 왔어요")

        # [흐름] 버튼을 눌러야만 아래 leave_one_out()이 실행된다(입력마다 자동 실행하면 낭비)
        if st.button("분석"):
            if not text.strip():
                st.warning("분석할 문장을 입력하세요.")
            else:
                # [왜] try/except — 이번에도 실패가 조용히 앱을 죽이지 않고 원인을 화면에 남긴다.
                try:
                    # [흐름] 단어 수+1번 추론이 걸리므로 spinner로 "처리 중"임을 알린다
                    with st.spinner("단어를 하나씩 빼보는 중..."):
                        label, score, contributions = leave_one_out(text)
                except Exception as e:  # noqa: BLE001 — 원인을 화면에 드러내고 앱은 계속 동작
                    logging.exception("단어 기여도 분석 실패: %r", text)
                    st.warning(f"분석 실패: {e}")
                else:
                    # [흐름] 성공했을 때만 결과를 그린다(else는 예외가 없을 때만 실행)
                    emoji = "😊" if label == "긍정" else "😞"
                    st.markdown(f"**판정: {label} {emoji} (확신도 {score:.1%})**")
                    render_word_highlight(label, contributions)
                    st.caption("색이 진한 단어일수록 '그 단어를 빼면 판정이 크게 흔들렸다' = 기여도가 크다는 뜻입니다.")


# [주의] streamlit run이 이 파일을 직접 실행할 때만 main()이 호출된다 — import될 때는 UI가 뜨지 않는다
if __name__ == "__main__":
    main()
