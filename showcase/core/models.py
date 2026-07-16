# showcase/core/models.py — 교안 앱(apps/*)의 모델 로직을 재사용하는 얇은 로더 계층.
# [왜] 중복 구현 금지(계획 규율) — apps/m3·m4의 함수를 그대로 import해 showcase가 소비한다.
#      showcase는 "같은 모델을 에이전트의 tool로 지휘"하는 레이어일 뿐, 모델 로직을 다시 짜지 않는다.
import pandas as pd
import seaborn as sns
import streamlit as st

# [Codex] apps.m3_penguins는 import만 해도 matplotlib rcParams(font·unicode_minus)를 전역 변경한다.
#   models.py는 모든 showcase 페이지가 import하므로, 감성 페이지처럼 펭귄과 무관한 화면까지 그 부작용을
#   딸려 받았다. → m3 import를 실제 사용 시점(get_penguin_model)으로 지연해 부작용을 격리한다.
#   (m4_image·m4_sentiment는 모듈레벨 부작용이 없어 상단 import 유지.)
# [왜] `streamlit run apps/파일.py`는 apps/ 폴더만 import 경로에 넣는다 — `from apps...`가 되려면
#      프로젝트 루트(이 파일의 두 단계 상위(프로젝트 루트))를 경로에 직접 추가해야 어떤 실행 방식·위치에서도 동작한다.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apps.m4_image import load_model as _load_image_model
from apps.m4_sentiment import analyze as _analyze_sentiment
from apps.m4_sentiment import load_pipeline as _load_sentiment_pipeline


@st.cache_resource
def get_penguin_model():
    """기본 하이퍼파라미터로 학습한 펭귄 분류기를 재사용한다. 반환: (model, LabelEncoder, feature 목록)."""
    from apps.m3_penguins import FEATURES, train_model  # 지연 import(matplotlib 전역변경 격리)

    df = sns.load_dataset("penguins").dropna()
    out = train_model(df, model_type="RandomForest", n_estimators=100, max_depth=5)
    return out["model"], out["le"], FEATURES


def predict_penguin(features: dict) -> dict:
    """부리·물갈퀴·체중 dict → {species, proba}. 에이전트 tool(classify_penguin)이 호출한다."""
    model, le, feats = get_penguin_model()
    X = pd.DataFrame([[features[f] for f in feats]], columns=feats)
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    return {"species": str(le.classes_[idx]), "proba": float(proba[idx])}


@st.cache_resource
def get_image_model():
    """MobileNetV2 (model, preprocess, categories)를 재사용한다 (apps/m4_image.load_model)."""
    return _load_image_model()


def predict_image(pil_image) -> list:
    """PIL 이미지 → top-5 [(클래스명, 확률)]. 에이전트 tool(classify_image)이 호출한다."""
    import torch  # [왜] 지연 import — m4_image lazy화로 상단 torch가 없다. 추론 시점에만 로드한다.

    model, preprocess, categories = get_image_model()
    x = preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        logits = model(x)
    probs = torch.nn.functional.softmax(logits[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    return [(categories[int(i)], float(p)) for i, p in zip(top5_idx, top5_prob)]


def get_sentiment_pipeline():
    """KoELECTRA 감성분석 pipeline을 재사용한다 (apps/m4_sentiment.load_pipeline)."""
    return _load_sentiment_pipeline()


def predict_sentiment(text: str) -> dict:
    """문장 → {label: '긍정'/'부정', score}. apps/m4_sentiment.analyze의 라벨 매핑(1→긍정)을 그대로 재사용."""
    return _analyze_sentiment(text)


@st.cache_resource
def get_heart_model():
    """심장병 분류기(RandomForest)·test 정확도·피처 중앙값을 재사용한다 (apps/heart_disease.train_model)."""
    from apps.heart_disease import FEATURES, load_data, train_model  # 지연 import(matplotlib rcParams 전역변경 격리)

    df = load_data()
    model, test_acc, _ = train_model(df)
    return model, test_acc, df[FEATURES].median().to_dict()


def predict_heart(features: dict) -> dict:
    """주요 지표 dict → {proba, test_acc, filled}. 에이전트 tool(predict_heart)이 호출한다.
    [왜] 13개 지표를 전부 물으면 에이전트 대화가 설문이 된다 — 미입력 지표는 데이터 중앙값으로
    채우고, 무엇을 채웠는지(filled)를 함께 돌려줘 사용자가 근사치임을 알게 한다."""
    from apps.heart_disease import FEATURES

    model, test_acc, medians = get_heart_model()
    row = {f: features.get(f, medians[f]) for f in FEATURES}
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    proba = float(model.predict_proba(X)[0][1])
    return {"proba": proba, "test_acc": test_acc, "filled": [f for f in FEATURES if f not in features]}
