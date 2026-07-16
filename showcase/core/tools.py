# showcase/core/tools.py — 3개 ML 모델을 에이전트의 도구(tool)로 노출한다.
# [흐름] 20강 function-calling 규약: OpenAI 포맷 스키마 + finish_reason=="tool_calls" 루프에서 run_tool 디스패치.
# [설계] calculate 같은 튜토리얼 tool은 두지 않는다(M4/P2) — 3 ML tool로 깔끔하게.
import logging

from showcase.core.models import predict_image, predict_penguin, predict_sentiment

# --- OpenAI function-calling 스키마 ---
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_penguin",
            "description": "부리 길이/깊이, 물갈퀴 길이(mm)와 체중(g)으로 팔머펭귄 종(Adelie/Chinstrap/Gentoo)을 분류한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_length_mm": {"type": "number", "description": "부리 길이(mm)"},
                    "bill_depth_mm": {"type": "number", "description": "부리 깊이(mm)"},
                    "flipper_length_mm": {"type": "number", "description": "물갈퀴 길이(mm)"},
                    "body_mass_g": {"type": "number", "description": "체중(g)"},
                },
                "required": ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "한국어 문장의 긍정/부정 감성을 분석한다.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "분석할 한국어 문장"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_image",
            # ⚠️ M4(P2): 이 tool은 인자가 없고, 사용자가 방금 업로드한 이미지를 session 사이드채널로 받는다.
            #    PIL 이미지를 JSON 인자로 넘길 수 없어서 생긴 타협(누수형 추상화) — 에이전트가 이미지를
            #    직접 '선택'하지는 않는다는 한계를 README/데모에 명시한다.
            "description": "사용자가 방금 업로드한 이미지를 ImageNet 1000종으로 분류한다(인자 없음 — 최근 업로드 이미지를 사용).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _classify_penguin(args: dict, session_image=None) -> str:
    out = predict_penguin(args)
    return f"예측 종: {out['species']} (확신도 {out['proba']:.1%})"


def _analyze_sentiment(args: dict, session_image=None) -> str:
    # [Codex] 빈 문자열이면 모델을 호출하지 않는다 — 페이지에서 막아도 에이전트 경로엔 남아 있던 구멍.
    text = (args.get("text") or "").strip()
    if not text:
        return "분석할 문장이 비어 있습니다 — 문장을 입력해주세요."
    out = predict_sentiment(text)
    return f"감성: {out['label']} (확신도 {out['score']:.1%})"


def _classify_image(args: dict, session_image=None) -> str:
    if session_image is None:
        return "이미지가 업로드되지 않았습니다 — 먼저 이미지를 올려주세요."
    top5 = predict_image(session_image)
    return "top-5: " + ", ".join(f"{name}({p:.1%})" for name, p in top5)


TOOL_FUNCTIONS = {
    "classify_penguin": _classify_penguin,
    "analyze_sentiment": _analyze_sentiment,
    "classify_image": _classify_image,
}


def run_tool(name: str, args: dict, session_image=None) -> str:
    """도구를 실행하고 결과를 문자열로 반환. 예외는 삼키지 않고 로그로 남긴 뒤 문자열로 돌려준다
    (에이전트가 tool_result로 받아 다음 턴에서 처리하도록) — 배치/루프가 예외로 멈추지 않게."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"알 수 없는 도구: {name}"
    try:
        return fn(args, session_image=session_image)
    except Exception as e:  # noqa: BLE001 — tool 실패를 에이전트 루프로 문자열 전달
        logging.exception("tool 실행 실패: %s %r", name, args)
        return f"도구 '{name}' 실행 중 오류: {e}"
