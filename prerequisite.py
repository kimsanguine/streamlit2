"""Streamlit 특강 — 사전 다운로드 스크립트

[왜] ST3(딥러닝데모앱)에서 쓰는 MobileNetV2(13.6MB)와 KoELECTRA(56.5MB)를
    50명이 수업 시간에 동시에 내려받으면 강의실 네트워크가 버티지 못한다.
    (ViT-base 346MB급 모델을 실수로 동시에 받으면 수업이 통째로 멈춘다 — 실측 경고)
    이 스크립트를 수업 전날 각자 PC에서 한 번 실행해두면, 당일에는 로컬 캐시에서
    바로 불러오기만 하면 되므로 네트워크 병목이 사라진다.

실행: python3.11 prerequisite.py  (Windows: py -3.11 prerequisite.py)
"""

import time
import urllib.request


def download_mobilenet():
    print("=" * 60)
    print("[1/2] MobileNetV2 (ImageNet, 약 13.6MB) 다운로드/캐시 확인")
    print("=" * 60)
    t0 = time.time()
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)  # 캐시에 없으면 여기서 실제 다운로드가 일어난다
    model.eval()
    elapsed = time.time() - t0
    print(f"✅ MobileNetV2 준비 완료 — {elapsed:.1f}초 (클래스 {len(weights.meta['categories'])}개)")
    print("   저장 위치: ~/.cache/torch/hub/checkpoints/\n")


def download_koelectra():
    print("=" * 60)
    print("[2/2] KoELECTRA-small NSMC (한국어 감성분석, 약 56.5MB) 다운로드/캐시 확인")
    print("=" * 60)
    t0 = time.time()
    from transformers import pipeline

    clf = pipeline("sentiment-analysis", model="daekeun-ml/koelectra-small-v3-nsmc")
    # 다운로드뿐 아니라 실제 추론 1회까지 돌려서 "정말 쓸 수 있는 상태"인지 확인한다
    result = clf("사전 다운로드 테스트 문장입니다")
    elapsed = time.time() - t0
    print(f"✅ KoELECTRA 준비 완료 — {elapsed:.1f}초")
    print(f"   테스트 추론 결과: {result}")
    print("   [주의] label은 'positive'/'negative'가 아니라 '1'(긍정)/'0'(부정)입니다 —")
    print("          m4_sentiment.py가 이 형식에 맞춰 라벨을 매핑합니다.")
    print("   저장 위치: ~/.cache/huggingface/hub/\n")


def check_ollama_embedding():
    print("=" * 60)
    print("[선택] ST4 로컬 임베딩 — Ollama nomic-embed-text 확인")
    print("=" * 60)
    # [왜] nomic-embed-text는 torch/transformers처럼 파이썬 코드로 받는 모델이 아니라
    #      Ollama CLI(`ollama pull`)로 받는 별도 모델이다 — 이 스크립트는 다운로드를
    #      대신해주지 않고, 이미 받아뒀는지만 HTTP로 확인해 안내한다.
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as resp:
            import json as _json

            tags = _json.load(resp)
            names = [m.get("name", "") for m in tags.get("models", [])]
            if any("nomic-embed-text" in n for n in names):
                print("✅ nomic-embed-text 준비 완료 — ST4 PDF RAG 챗봇에서 로컬 임베딩을 바로 쓸 수 있습니다.")
            else:
                print("⚠️ Ollama는 실행 중이지만 nomic-embed-text가 없습니다 — 아래 명령으로 받아두세요(선택, 약 274MB):")
                print("   ollama pull nomic-embed-text")
    except Exception:
        print("ℹ️ Ollama 서버가 응답하지 않습니다 — 로컬 임베딩은 선택 사항입니다.")
        print("   써보고 싶다면 Ollama 설치 후 아래 한 줄을 받아두세요:")
        print("   ollama pull nomic-embed-text")
    print("   (안 받아도 openrouter·openai 임베딩이나 TF-IDF 폴백으로 ST4를 그대로 따라갈 수 있습니다.)\n")


if __name__ == "__main__":
    print("Streamlit — ST3 딥러닝데모앱 사전 다운로드를 시작합니다.")
    print("예상 소요 시간: 처음 실행 시 약 1~2분(모델 다운로드), 이미 캐시돼 있으면 수 초 이내.\n")

    total_t0 = time.time()
    download_mobilenet()
    download_koelectra()
    check_ollama_embedding()
    total_elapsed = time.time() - total_t0

    print("=" * 60)
    print(f"🎉 모든 모델 준비 완료 — 총 {total_elapsed:.1f}초")
    print("   내일 수업 당일에는 이 스크립트를 다시 실행해도 캐시 덕분에 몇 초 안에 끝납니다.")
    print("=" * 60)
