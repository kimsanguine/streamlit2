# ST3 — 딥러닝 데모 ①: 이미지 분류 (MobileNetV2) + "가려보기" 히트맵
# [왜] 13강에서 CNN을 직접 손으로 쌓아봤다면, 여기서는 이미 학습된 CNN(MobileNetV2,
#      ImageNet 1000종)을 웹 UI에 얹어 "누구나 브라우저에서 바로 써볼 수 있는" 완제품으로 만든다.
# [흐름] MobileNetV2 13.6MB · CPU 추론 약 87ms(실측) — 강의실 노트북에서도 버벅이지 않는 크기.
# [주의] UI는 main()으로 감싼다 — showcase(core.models)가 load_model을 import해 재사용하므로,
#        모듈 레벨에 st.* UI를 두면 import만 해도 이 화면이 딸려 나온다(라이브러리는 import-safe해야 함).
# 실행: python3.11 -m streamlit run apps/m4_image.py

import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# [왜] torch·torchvision은 import만 3초+ — 파일 최상단에 두면 앱이 뜨는 것부터 느려진다.
#      실제 쓰는 함수(load_model·히트맵·추론) 안에서 import해 화면은 즉시 뜨게 한다(lazy import).

# [왜] matplotlib 기본 폰트(DejaVu Sans)엔 한글 글리프가 없어 히트맵 제목이 □□로 깨진다.
#      로컬은 Mac(AppleGothic)이 있어 멀쩡하지만, Streamlit Cloud(Linux)엔 packages.txt의
#      fonts-nanum으로 NanumGothic을 설치해야 한다(ST2 m3_penguins.py와 동일한 패턴).
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
    if any(_font.name == _f for _font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# [왜] 격자 크기 — 너무 작으면(예: 16x16=256번) 배치 추론이 느려지고, 너무 크면(예: 4x4) 위치 정보가 뭉개진다.
GRID = 8  # 8x8=64칸


# [왜] cache_resource — 모델·전처리기·클래스 목록은 "리소스"다. rerun마다 다시 로드하면
#      13.6MB 가중치를 매번 새로 읽어야 한다. ST2에서 배운 cache_resource가 여기서 실전 투입된다.
@st.cache_resource
def load_model():
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # lazy — 최상단 주석 참고
    # ImageNet으로 사전학습된 공식 가중치
    weights = MobileNet_V2_Weights.DEFAULT
    # 가중치가 이미 들어있는 상태로 모델 생성(처음부터 학습 X)
    model = mobilenet_v2(weights=weights)
    model.eval()  # [왜] eval() — Dropout/BatchNorm을 "추론 모드"로 고정. 학습 때와 동작이 다르다.
    preprocess = weights.transforms()  # 학습 때와 동일한 리사이즈·정규화를 자동으로 맞춰줌
    categories = weights.meta["categories"]  # ImageNet 1000종 클래스 이름 목록
    return model, preprocess, categories


# [왜] occlusion(가려보기) — 이미지를 8x8 격자로 나눠 한 칸씩 회색으로 가려보고, 원래 top-1
#      클래스의 확률이 얼마나 떨어지는지 측정한다. 확률이 많이 떨어진 칸일수록 모델이 "그
#      부분을 보고 판단했다"는 뜻이다. gradient(역전파) 없이 순수 추론 반복만으로 설명력을 얻는
#      가장 단순한 형태의 설명가능AI(XAI, explainable AI) 기법이다 — 노트북 3절에서 먼저 확인한
#      바로 그 로직을 그대로 가져온다.
def compute_occlusion_heatmap(image: Image.Image, model, preprocess, grid: int = GRID):
    """반환: (모델이 실제로 본 224x224 크롭, grid x grid 중요도 배열, top1 클래스 idx, top1 확률)"""
    import torch  # lazy — 최상단 주석 참고
    import torchvision.transforms.functional as TF
    # [왜] preprocess()가 내부에서 하는 resize+crop을 미리 적용해두면, 화면에 그릴 이미지와
    #      모델이 실제로 보는 이미지가 같은 좌표계를 가져 히트맵이 사진 위에 정확히 겹쳐진다.
    view = TF.center_crop(
        TF.resize(image, preprocess.resize_size, interpolation=TF.InterpolationMode.BILINEAR),
        preprocess.crop_size,
    )
    # [흐름] PIL 이미지 → numpy 배열(H,W,3) — 픽셀 값을 칸 단위로 잘라 가리려면 배열이 필요
    arr = np.array(view)
    # [흐름] 정사각형 크롭(224x224)을 grid로 나눈 몫이 "한 칸의 변 길이"
    patch = arr.shape[0] // grid  # 한 칸의 픽셀 크기(224÷8=28px)

    # [왜] preprocess()를 "resize+crop"과 "정규화"로 나눠 다시 조립한 버전 — 64번 반복 호출할
    #      때마다 resize+crop까지 다시 하면 낭비이므로, 정규화만 반복한다.
    def normalize(pil_img):
        return TF.normalize(TF.to_tensor(pil_img), preprocess.mean, preprocess.std)

    with torch.inference_mode():  # 원본(가리지 않은) 이미지의 기준 확률 — 이걸 기준으로 "얼마나 떨어졌는지" 잰다
        base_logits = model(normalize(view).unsqueeze(0))
    base_probs = torch.nn.functional.softmax(base_logits[0], dim=0)
    top1_prob, top1_idx = torch.topk(base_probs, 1)  # 1등 클래스 하나만 필요(히트맵은 "그 클래스" 기준)
    baseline_prob = float(top1_prob[0])

    # [흐름] 64장(한 칸씩 가린 이미지)을 하나의 배치로 묶어 "배치 1회 추론"으로 처리한다 —
    #        64번 따로 부르는 것보다 훨씬 빠르다(실측 약 5~6초, CPU 기준).
    batch = []
    # 세로 칸 인덱스(0~7)
    for gy in range(grid):
        for gx in range(grid):  # 가로 칸 인덱스(0~7) — gy,gx 조합으로 64칸을 모두 순회
            occluded = arr.copy()  # [주의] 원본 arr을 직접 바꾸면 다음 칸 계산이 오염된다 — 매번 복사
            occluded[gy * patch:(gy + 1) * patch, gx * patch:(gx + 1) * patch] = 127  # 회색으로 마스킹
            batch.append(normalize(Image.fromarray(occluded)))
    with torch.inference_mode():  # 64장을 한 번에 추론 — for문으로 64번 따로 부르지 않는다
        occ_logits = model(torch.stack(batch))
    occ_probs = torch.nn.functional.softmax(occ_logits, dim=1)[:, top1_idx[0]]  # 각 칸을 가렸을 때 top1 클래스의 확률

    # [왜] 확률이 많이 떨어질수록(=중요도 높음) 뜨거운 색으로 표시할 값 — 오히려 확률이 오른
    #      칸(음수)은 0으로 잘라 "중요했던 영역"만 강조한다.
    importance = np.clip(baseline_prob - occ_probs.reshape(grid, grid).numpy(), 0, None)
    # [흐름] view(화면용 이미지)·importance(8x8 중요도)·top1 클래스·기준 확률 4가지를 그대로 반환
    return view, importance, top1_idx.item(), baseline_prob


def main():
    # [왜] set_page_config는 스크립트에서 딱 한 번, 다른 st.* 호출보다 먼저 와야 한다(공식 규칙).
    st.set_page_config(page_title="이미지 분류 데모", page_icon="🖼️", layout="centered")
    # [흐름] 제목·부제 — 이 앱이 무엇인지 3초 안에 알려준다
    import torch  # lazy — 최상단 주석 참고(추론 블록에서 사용)
    st.title("🖼️ MobileNetV2 이미지 분류 데모")
    st.caption("13강 CNN의 '완제품 버전' — ImageNet 1000종 분류, CPU 추론 약 87ms(실측)")

    # [흐름] 캐시되어 있으면 즉시 반환(재로드 없음)
    model, preprocess, categories = load_model()

    st.subheader("이미지를 올려주세요")
    uploaded = st.file_uploader("파일 선택", type=["jpg", "jpeg", "png"])  # 반환값은 UploadedFile 또는 None
    # [주의] camera_input은 브라우저가 웹캠 권한을 요구한다. 강의실 PC·네트워크 정책상 권한이
    #        막히면 camera가 None으로 남을 뿐 에러는 나지 않는다 — file_uploader만으로도 데모에 지장 없다.
    camera = st.camera_input("또는 사진 촬영 (웹캠 권한이 막히면 위 파일 업로드를 사용하세요)")

    # ✏️ [학생 실습 지점] uploaded가 없으면 camera를, 둘 다 없으면 None을 쓰는 우선순위 표현식
    source = uploaded or camera

    if source is not None:
        # [흐름] uploaded든 camera든 여기서부터는 완전히 같은 처리(같은 image 변수)
        image = Image.open(source).convert("RGB")  # [주의] PNG(RGBA)·흑백도 항상 3채널로 맞춰 모델 입력 오류를 막는다
        st.image(image, caption="입력 이미지", width="stretch")

        with st.spinner("분류 중..."):
            # 추론 시간 측정 시작 — "몇 ms 걸렸는지"를 화면에 보여주기 위해
            t0 = time.time()
            x = preprocess(image).unsqueeze(0)  # [흐름] (3,224,224) → (1,3,224,224) 배치 차원 추가
            st.caption(
                f"정규화 후 값 범위: {x.min():.2f} ~ {x.max():.2f} "
                "— 0~255 정수였던 픽셀이 음수 포함 float로 바뀐 게 정규화입니다"
            )
            # [왜] inference_mode — no_grad보다 한 단계 더 강하게 autograd 추적을 꺼서
            #      추론 전용 코드에서 메모리·속도 이득이 더 크다(PyTorch 공식 권장).
            with torch.inference_mode():
                logits = model(x)
            # [왜] softmax — 로짓(임의 실수, 클래스 선호도)을 "합이 1인 확률"로 바꿔야 사람이 읽을 수 있다.
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            top5_prob, top5_idx = torch.topk(probs, 5)
            elapsed = time.time() - t0  # 추론 시간 측정 끝

        # [흐름] 걸린 시간을 사용자에게 보여줘 "체감 속도"를 숫자로 확인시킨다
        st.success(f"분류 완료 ({elapsed * 1000:.0f}ms)")

        # [왜] set_index("클래스") — st.bar_chart는 인덱스를 x축(막대 이름)으로 쓴다.
        #      ST1에서 배운 "인덱스가 곧 x축"이라는 규칙이 여기서도 그대로 적용된다.
        result_df = pd.DataFrame(
            {
                "클래스": [categories[i] for i in top5_idx],
                "확률": [float(p) for p in top5_prob],
            }
        ).set_index("클래스")
        # top-5 확률을 막대그래프로 — 숫자표보다 "1등이 압도적인지 근소한지"가 한눈에 들어온다
        st.bar_chart(result_df)

        st.divider()  # top-5 결과와 히트맵 구간을 시각적으로 구분
        st.subheader("🔍 모델이 어디를 보고 판단했을까?")
        st.caption(
            "8×8 칸으로 나눠 한 칸씩 회색으로 가려보고, top-1 확률이 얼마나 떨어지는지 측정합니다 "
            "— 확률이 많이 떨어진 칸일수록 '그 부분을 보고 판단했다'는 뜻입니다."
        )
        # [왜] 버튼 뒤에 둔다 — 64칸 배치 추론(약 5~6초)을 top-5 분류마다 매번 자동 실행하면
        #      평소 사용이 느려진다. "보고 싶을 때만" 계산하는 게 JIT 원칙(ST2 caching 절과 같은 정신).
        if st.button("🧩 가려보기 히트맵 계산 (5~6초 소요)"):  # 버튼을 눌러야만 아래 블록이 실행된다
            with st.spinner("64칸을 하나씩 가려보는 중... (배치 1회 추론)"):
                view, importance, top1_i, baseline_prob = compute_occlusion_heatmap(image, model, preprocess)

            fig, ax = plt.subplots(figsize=(6, 6))  # 정사각형 사진에 맞춰 가로세로 비율도 정사각형으로
            ax.imshow(view)  # 배경 — 모델이 실제로 본 224x224 크롭 사진
            # [흐름] 8x8(64칸)짜리 저해상도 중요도 배열을 224x224 사진 위에 반투명(alpha=0.55)으로
            #        겹친다 — interpolation="bilinear"가 8x8을 부드럽게 확대해 자연스러운 히트맵을 만든다.
            ax.imshow(
                importance, cmap="hot", alpha=0.55,
                extent=(0, view.width, view.height, 0), interpolation="bilinear",
            )
            ax.axis("off")  # 좌표축 눈금은 히트맵 해석에 필요 없어 숨긴다
            # [흐름] 제목에 top-1 클래스·확률을 같이 넣어 히트맵과 숫자를 한 화면에서 대조하게 한다
            ax.set_title(f"top-1: {categories[top1_i]} ({baseline_prob * 100:.1f}%)")
            # [주의] Streamlit은 matplotlib Figure를 자동으로 그려주지 않는다 — st.pyplot()으로 직접 넘겨야 한다
            st.pyplot(fig)

            # [왜] 랜덤 베이스라인 — "확률 9.7%가 낮아 보이지만 실은 크다"는 감각을
            #      숫자로 확인시키기 위한 단순 참고치(1000종 중 무작위로 하나를 찍었을 때의 확률).
            #      정답 여부는 알 수 없으므로(ground truth 없음) '예측 클래스의 확률'이라고만 부른다.
            random_baseline = 1 / len(categories)
            st.caption(
                f"예측 확률 {baseline_prob * 100:.1f}%는 무작위 참고치({random_baseline * 100:.1f}%)의 "
                f"{baseline_prob / random_baseline:.0f}배 — 확률 자체는 낮아 보여도 무작위보다 훨씬 높습니다."
            )
            # [왜] 정직 각주 — 이 기법의 한계를 먼저 말해두지 않으면 학생들이 "이게 정답"이라고
            #      과신하게 된다. expander로 접어둬 필요한 사람만 펼쳐 보게 한다.
            with st.expander("⚠️ 정직한 한계 — occlusion 히트맵이 말해주는 것과 못 말해주는 것"):
                st.markdown(
                    "이 히트맵은 **occlusion(가려보기)** 기법입니다 — 실제로 칸을 가려보고 확률이 "
                    "얼마나 떨어지는지 재는 perturbation 계열 XAI 방법으로, 모델 내부를 몰라도 되지만 "
                    f"격자 수({GRID}x{GRID})만큼 추론이 필요하고 격자 해상도만큼만 정보가 있어 경계가 "
                    "뭉툭합니다. 다른 계열인 **Grad-CAM**은 gradient(역전파) 1회로 빠르게 히트맵을 "
                    "만들지만, 해상도가 마지막 conv feature map 크기에 묶입니다 — 서로 원리가 다른 "
                    "두 방법일 뿐 어느 쪽이 항상 더 정확한 것은 아닙니다. 심화로 Grad-CAM(예: "
                    "`torchcam` 라이브러리)을 찾아보세요."
                )
    else:
        # [흐름] 아직 이미지가 없을 때의 안내 — 빈 화면 대신 "무엇을 해야 하는지"를 알려준다
        # [주의] source가 None이면 이 분기로 온다 — uploaded·camera 둘 다 비어 있는 상태
        st.info("이미지를 업로드하거나 촬영하면 top-5 예측과 '어디를 보고 판단했는지' 히트맵이 여기 표시됩니다.")


if __name__ == "__main__":
    main()  # [주의] streamlit run이 이 파일을 직접 실행할 때만 UI가 뜬다 — import될 때는 안 뜬다
