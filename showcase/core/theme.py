"""쇼케이스 디자인 시스템 — 라이트/다크 테마 + 재사용 UI 헬퍼.

순수 UI 헬퍼 모듈이다. 모델/에이전트 로직(core.models, core.agent, core.tools)을
import하지 않으며, Streamlit 컴포넌트 렌더링만 담당한다.

호출 순서: 각 페이지 최상단에서 ``apply_theme()``을 먼저 호출해 CSS를 주입한 뒤,
``theme_toggle()``·``hero()``·``kpi_row()``·``card()``를 사용한다.

테마 상태:
- ``st.session_state.theme``: 항상 "light" 또는 "dark" 문자열(초기값 "light").
  [Codex] config.toml이 base=light로 고정돼 서버는 실제 화면 색을 알 수 없다. 이전의 auto 모드
  (``prefers-color-scheme`` 미디어쿼리)는 segmented control 상태와 화면이 불일치하는 버그를 낳아
  제거했다 — 이제 항상 명시 테마만 쓴다. OS 다크 사용자는 토글로 다크를 선택한다.
"""

import streamlit as st

# 셀 1/4: 디자인 토큰 (light/dark 공통 + 개별)
_SHARED_VARS = """
  --sc-accent-1: #7C3AED;
  --sc-accent-2: #06B6D4;
  --sc-accent-grad: linear-gradient(135deg, var(--sc-accent-1), var(--sc-accent-2));
  --sc-radius: 16px;
  --sc-radius-sm: 12px;
  --sc-font-sans: 'Pretendard Variable', Pretendard, -apple-system, sans-serif;
  --sc-font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
"""

_LIGHT_VARS = """
  --sc-bg: #F3F4F7;
  --sc-bg-elevated: #FFFFFF;
  --sc-text: #0F172A;
  --sc-text-muted: #4B5876;
  --sc-border: rgba(15, 23, 42, 0.12);
  --sc-glass-bg: rgba(255, 255, 255, 0.72);
  --sc-glass-border: rgba(15, 23, 42, 0.10);
  --sc-accent-soft: rgba(124, 58, 237, 0.10);
  --sc-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
  --sc-mesh-opacity: 0.14;
"""

_DARK_VARS = """
  --sc-bg: #0B0E14;
  --sc-bg-elevated: #161B26;
  --sc-text: #E6EDF3;
  --sc-text-muted: #9AA7B8;
  --sc-border: rgba(230, 237, 243, 0.11);
  --sc-glass-bg: rgba(255, 255, 255, 0.045);
  --sc-glass-border: rgba(230, 237, 243, 0.10);
  --sc-accent-soft: rgba(124, 58, 237, 0.20);
  --sc-shadow: 0 10px 34px rgba(0, 0, 0, 0.5);
  --sc-mesh-opacity: 0.18;
"""

# 셀 2/4: 웹폰트 @import — [Codex] @import는 반드시 <style> 최상단(다른 규칙 앞)에 와야 유효하다.
#   :root 규칙 뒤에 두면 CSS 규격상 무시돼 Pretendard·JetBrains Mono가 다운로드되지 않고 fallback으로 렌더된다.
_FONT_IMPORTS = (
    "@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');\n"
    "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');"
)

# 정적 CSS (컴포넌트 스타일 — 테마 무관, var(--sc-*)만 참조)
_STATIC_CSS = """
html, body { overflow-x: hidden; }

.stApp {
  background: var(--sc-bg) !important;
  color: var(--sc-text);
}
.stApp, .stApp p, .stApp span:not([data-testid="stIconMaterial"]), .stApp label, .stApp div {
  font-family: var(--sc-font-sans);
}
h1, h2, h3, h4, h5, h6 { font-family: var(--sc-font-sans); color: var(--sc-text); }

/* [버그방지] Streamlit Material 아이콘(사이드바 접기·채팅 아바타 face/smart_toy 등)은 폰트
   오버라이드에서 제외한다 — span 전체에 Pretendard를 걸면 아이콘이 글리프 대신 리거처 원문
   텍스트("face"·"smart_toy"·"keyboard_double_arrow_left")로 샌다(라이브 관측으로 확인). */
[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded' !important;
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: var(--sc-mesh-opacity);
  background:
    radial-gradient(circle at 15% 20%, var(--sc-accent-1), transparent 45%),
    radial-gradient(circle at 85% 0%, var(--sc-accent-2), transparent 40%),
    radial-gradient(circle at 60% 90%, var(--sc-accent-1), transparent 35%);
  filter: blur(60px);
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
  background: var(--sc-glass-bg);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border-right: 1px solid var(--sc-glass-border);
}
[data-testid="stSidebar"] * { color: var(--sc-text); }

.stButton > button {
  border-radius: var(--sc-radius);
  border: 1px solid var(--sc-glass-border);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: var(--sc-shadow);
}
.stButton > button[kind="primary"] {
  background: var(--sc-accent-grad);
  border: none;
  color: #fff;
}

:focus-visible { outline: 2px solid var(--sc-accent-2); outline-offset: 2px; }
.stApp img { max-width: 100%; height: auto; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--sc-glass-border); border-radius: 999px; }

/* 제네릭 Streamlit chrome 숨김 (무지개 데코·러닝 상태 위젯) — '튜토리얼 티' 제거 */
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* 입력 위젯 테밍 — 다크 캔버스에서 흰 박스로 남지 않게 표면·보더를 토큰에 맞춘다.
   [주의] config.toml이 base=light라 BaseWeb 위젯 내부는 라이트로 렌더된다 → data-baseweb으로
   배경을 강제(다크 전략 C). Streamlit 버전업 시 data-baseweb 구조가 바뀌면 재확인 필요(fragile). */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stChatInput"] textarea {
  background: var(--sc-bg-elevated) !important;
  color: var(--sc-text) !important;
  caret-color: var(--sc-accent-1);
}
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="base-input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
  background: var(--sc-bg-elevated) !important;
  border: 1px solid var(--sc-border) !important;
  border-radius: var(--sc-radius-sm) !important;
}
[data-testid="stNumberInput"] button {
  background: var(--sc-bg-elevated) !important;
  color: var(--sc-text-muted) !important;
  border-color: var(--sc-border) !important;
}
/* 위젯 라벨 대비 — 다크에서 라벨이 dark-on-dark로 소멸하는 것 방지 (p만 타깃, span 금지) */
[data-testid="stWidgetLabel"] p { color: var(--sc-text) !important; font-weight: 600; }

/* st.metric — config base=light가 값/라벨을 다크 텍스트로 강제 → 다크 캔버스에서 소멸. 토큰으로 복구.
   (결과 패널의 핵심 앵커라 반드시 보여야 함) */
[data-testid="stMetricValue"] { color: var(--sc-text) !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: var(--sc-text-muted) !important; }

/* st.chat_message 본문 — config textColor(#0F172A)가 다크 캔버스에서 소멸(dark-on-dark).
   채팅에 스코프(커스텀 sc-* HTML은 건드리지 않음). 라이트에선 값이 같아 무해. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
  color: var(--sc-text) !important;
}

/* [Codex] 다크에서 라이트로 남던 native surface/텍스트 보정 — file_uploader / page_link.
   색 위주로 최소 개입(아이콘 폰트는 건드리지 않음). */
[data-testid="stFileUploaderDropzone"] {
  background: var(--sc-bg-elevated) !important;
  border-color: var(--sc-border) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] small,
[data-testid="stPageLink"] p { color: var(--sc-text) !important; }

/* [Codex] KPI 값 mono 폰트 — .stApp div(specificity 0,1,1)가 .sc-kpi-value(0,1,0)를 덮어
   sans로 나오던 것을 타일 스코프(0,2,0)로 복구. */
.sc-kpi-tile .sc-kpi-value { font-family: var(--sc-font-mono); }
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
[data-testid="stChatInput"]:focus-within {
  border-color: var(--sc-accent-2) !important;
  box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.22) !important;
}

/* 기본 알림(st.info/warning/success) → 글래스 리스킨 (플랫 색이 커스텀 미학과 충돌) */
[data-testid="stAlert"] {
  background: var(--sc-glass-bg) !important;
  border: 1px solid var(--sc-glass-border) !important;
  border-left: 3px solid var(--sc-accent-2) !important;
  border-radius: var(--sc-radius-sm) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

/* st.container(border=True) — 결과 패널 글래스 통일 */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--sc-radius) !important;
}

/* 셀링 배지 (핵심 셀링포인트 풀폭 강조) */
.sc-badge {
  display: flex; align-items: center; gap: 0.7rem;
  padding: 0.85rem 1.15rem;
  margin-bottom: 1.5rem;
  border-radius: var(--sc-radius);
  background: var(--sc-accent-soft);
  border: 1px solid var(--sc-glass-border);
  border-left: 4px solid var(--sc-accent-1);
}
.sc-badge__icon { font-size: 1.3rem; line-height: 1; }
.sc-badge__text { font-size: 0.98rem; line-height: 1.5; color: var(--sc-text); }
.sc-badge__text strong { color: var(--sc-text); }

/* 한글 줄바꿈 — '동작' 같은 단어가 홀로 다음 줄에 떨어지는 어색한 wrap 방지 */
.sc-hero__subtitle, .sc-card__body, .sc-badge__text { word-break: keep-all; overflow-wrap: anywhere; }

/* 히어로 로드 연출 (idle 제거) */
@keyframes sc-fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.sc-hero__rule { animation: sc-fade-up 0.5s ease both; }
.sc-hero__title { animation: sc-fade-up 0.5s ease both; animation-delay: 0.06s; }
.sc-hero__subtitle { animation: sc-fade-up 0.5s ease both; animation-delay: 0.12s; }
@media (prefers-reduced-motion: reduce) {
  .sc-hero__rule, .sc-hero__title, .sc-hero__subtitle { animation: none; }
}

/* 히어로 */
.sc-hero {
  position: relative;
  overflow: hidden;
  padding: 3rem 2rem 2.5rem;
  margin-bottom: 1.5rem;
  border-radius: var(--sc-radius);
  background: var(--sc-glass-bg);
  border: 1px solid var(--sc-glass-border);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: var(--sc-shadow);
}
.sc-hero__blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(50px);
  opacity: 0.35;
  pointer-events: none;
}
.sc-hero__blob--1 { width: 260px; height: 260px; top: -80px; right: -60px; background: var(--sc-accent-1); }
.sc-hero__blob--2 { width: 220px; height: 220px; bottom: -100px; left: -40px; background: var(--sc-accent-2); }
.sc-hero__rule {
  position: relative; z-index: 1;
  width: 56px; height: 4px;
  border-radius: 999px;
  background: var(--sc-accent-grad);
  margin-bottom: 1rem;
}
.sc-hero__title {
  position: relative; z-index: 1;
  font-size: clamp(1.8rem, 4vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.25;
  margin: 0 0 0.5rem 0;
  color: var(--sc-text);
}
.sc-hero__subtitle {
  position: relative; z-index: 1;
  font-size: clamp(0.95rem, 1.6vw, 1.15rem);
  color: var(--sc-text-muted);
  max-width: 640px;
  margin: 0;
  line-height: 1.6;
}

/* KPI 카드 행 */
.sc-kpi-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.9rem;
  margin-bottom: 1.5rem;
}
.sc-kpi-tile {
  flex: 1 1 150px;
  min-width: 140px;
  padding: 1.1rem 1.2rem;
  border-radius: var(--sc-radius);
  background: var(--sc-glass-bg);
  border: 1px solid var(--sc-glass-border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--sc-shadow);
  position: relative;
  overflow: hidden;
}
.sc-kpi-tile::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--sc-accent-2);
}
.sc-kpi-value {
  font-family: var(--sc-font-mono);
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--sc-text);
  line-height: 1.2;
  word-break: break-word;
}
.sc-kpi-label {
  margin-top: 0.35rem;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--sc-text-muted);
}

/* 카드 */
.sc-card {
  padding: 1.4rem 1.5rem;
  border-radius: var(--sc-radius);
  background: var(--sc-glass-bg);
  border: 1px solid var(--sc-glass-border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--sc-shadow);
  margin-bottom: 1rem;
  max-width: 100%;
}
.sc-card__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px; height: 40px;
  border-radius: var(--sc-radius-sm);
  background: var(--sc-accent-soft);
  border: 1px solid var(--sc-glass-border);
  font-size: 1.2rem;
  margin-bottom: 0.7rem;
}
.sc-card__title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--sc-text);
  margin-bottom: 0.35rem;
}
.sc-card__body {
  font-size: 0.92rem;
  line-height: 1.6;
  color: var(--sc-text-muted);
  margin: 0;
}

/* 테마 토글 라벨 */
.sc-theme-toggle-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--sc-text-muted);
  margin-bottom: 0.4rem;
}

@media (max-width: 640px) {
  .sc-hero { padding: 2rem 1.25rem; }
  .sc-kpi-tile { flex: 1 1 100%; }
}
"""


def apply_theme() -> None:
    """세션 테마(라이트/다크) 기준 CSS를 주입한다. 초기값 라이트(config.toml base와 일치).

    [Codex] 이전엔 'auto' 모드에서 CSS ``prefers-color-scheme``로 OS 다크를 따랐지만, config.toml이
    base=light로 고정돼 서버는 실제 화면 색을 알 수 없었다 → segmented control(라이트 표시)과 실제
    화면(OS 다크)이 불일치하고, 선택 해제 시 컨트롤이 비는 버그가 있었다. auto를 제거하고 항상 명시
    테마를 쓴다(컨트롤=화면 일치). OS 다크 사용자는 토글로 다크를 선택한다.
    """
    theme = st.session_state.setdefault("theme", "light")
    chosen = _DARK_VARS if theme == "dark" else _LIGHT_VARS
    st.markdown(
        f"<style>\n{_FONT_IMPORTS}\n:root {{{_SHARED_VARS}}}\n:root {{{chosen}}}\n{_STATIC_CSS}\n</style>",
        unsafe_allow_html=True,
    )


# 셀 3/4: 테마 토글 + 히어로/KPI/카드 헬퍼
def _on_theme_change() -> None:
    """사용자가 토글을 조작하면 테마를 확정한다. 선택 해제(None)는 현재 테마로 복원해 컨트롤이 비지 않게."""
    sel = st.session_state.get("_theme_segmented")
    if sel is None:
        # [Codex] segmented_control은 선택된 항목 재클릭 시 None으로 해제됨 → 컨트롤 빈 상태 방지.
        st.session_state["_theme_segmented"] = (
            "🌙 다크" if st.session_state.get("theme") == "dark" else "🌤️ 라이트"
        )
        return
    st.session_state.theme = "dark" if "다크" in sel else "light"


def theme_toggle() -> None:
    """사이드바에 라이트/다크 토글을 렌더링한다."""
    current = st.session_state.get("theme", "light")
    options = ["🌤️ 라이트", "🌙 다크"]
    theme_to_label = {"light": options[0], "dark": options[1]}

    with st.sidebar:
        st.markdown('<div class="sc-theme-toggle-label">테마</div>', unsafe_allow_html=True)
        st.segmented_control(
            "테마",
            options,
            default=theme_to_label.get(current, options[0]),
            key="_theme_segmented",
            label_visibility="collapsed",
            on_change=_on_theme_change,
        )


def hero(title: str, subtitle: str = "") -> None:
    """큰 히어로 섹션(그라디언트 액센트 + 배경 블롭)을 렌더링한다."""
    subtitle_html = f'<p class="sc-hero__subtitle">{subtitle}</p>' if subtitle else ""
    # [주의] st.markdown은 4칸+ 들여쓰기 줄을 '코드블록'으로 해석한다 — HTML은 반드시 들여쓰기 없는
    #        단일 라인으로 넘겨야 raw 코드로 보이지 않고 실제로 렌더된다.
    html = (
        '<div class="sc-hero">'
        '<div class="sc-hero__blob sc-hero__blob--1"></div>'
        '<div class="sc-hero__blob sc-hero__blob--2"></div>'
        '<div class="sc-hero__rule"></div>'
        f'<h1 class="sc-hero__title">{title}</h1>'
        f"{subtitle_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def kpi_row(items: list[dict]) -> None:
    """가로 KPI 카드 행을 렌더링한다. items 각 원소는 {"label": str, "value": str}."""
    # 들여쓰기 없는 단일 라인 HTML(마크다운 코드블록화 방지).
    tiles = "".join(
        '<div class="sc-kpi-tile">'
        f'<div class="sc-kpi-value">{item.get("value", "")}</div>'
        f'<div class="sc-kpi-label">{item.get("label", "")}</div>'
        "</div>"
        for item in items
    )
    st.markdown(f'<div class="sc-kpi-row">{tiles}</div>', unsafe_allow_html=True)


def selling_badge(text_html: str, *, icon: str = "🔑") -> None:
    """핵심 셀링포인트를 전폭으로 강조하는 배지. text_html은 <strong> 등 인라인 HTML 허용."""
    # 들여쓰기 없는 단일 라인 HTML(마크다운 코드블록화 방지).
    html = (
        '<div class="sc-badge">'
        f'<div class="sc-badge__icon">{icon}</div>'
        f'<div class="sc-badge__text">{text_html}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def card(title: str, body: str = "", *, icon: str = "") -> None:
    """스타일된 카드 1개를 렌더링한다."""
    icon_html = f'<div class="sc-card__icon">{icon}</div>' if icon else ""
    body_html = f'<p class="sc-card__body">{body}</p>' if body else ""
    # 들여쓰기 없는 단일 라인 HTML(마크다운 코드블록화 방지).
    html = (
        '<div class="sc-card">'
        f"{icon_html}"
        f'<div class="sc-card__title">{title}</div>'
        f"{body_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# 셀 4/4: (의도적으로 비움) — 이 모듈은 UI 헬퍼만 export한다.
