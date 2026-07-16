import streamlit as st

from apps.m5b_agent_loop import PROVIDERS, provider_available
from apps.rag_chatbot import answer_with_context, assemble_context, retrieve
from apps.rag_lite import EMBED_PROVIDERS, _extract_pdf_text, chunk_text, embed_provider_available
from showcase.core.theme import apply_theme, hero, theme_toggle

st.set_page_config(page_title="RAG 챗봇", page_icon="📚", layout="wide")
apply_theme()
theme_toggle()
hero("📚 내 문서 RAG 챗봇", "문서 업로드 → 청킹 → 검색 → LLM 생성 답변 — 교안 rag_chatbot의 파이프라인을 그대로 재사용합니다")

# [왜] provider는 자동 감지 — 키/서버가 살아 있는 것을 골라 "질문 → 생성 답변"이 끊기지 않게 한다.
llm = next((p for p in PROVIDERS if provider_available(p)), None)
emb = next((p for p in EMBED_PROVIDERS if embed_provider_available(p)), None)
if llm:
    st.caption(f"✅ LLM: **{llm}** · 검색: **{emb or 'TF-IDF 폴백'}** — 검색된 근거로 답변을 생성합니다.")
else:
    st.caption("🧪 LLM 키/서버 없음 — 검색 데모 모드(관련 대목 표시)로 동작합니다.")

# [왜] st.stop() 대신 분기 구조 — 페이지가 import/bare 환경에서도 끝까지 안전하게 실행된다(교안 import-safe 규약).
uploaded = st.file_uploader("문서 업로드 (txt·md·pdf)", type=["txt", "md", "pdf"])
if uploaded is None:
    st.info("문서를 올리면 채팅으로 질문할 수 있습니다 — 여기 올린 문서는 🤖 에이전트의 `search_docs` 도구에서도 검색됩니다.")
else:
    if uploaded.name.lower().endswith(".pdf"):
        raw_text = _extract_pdf_text(uploaded.getvalue())
    else:
        raw_text = uploaded.getvalue().decode("utf-8", errors="ignore")

    if not raw_text.strip():
        st.warning("⚠️ 텍스트를 추출하지 못했습니다 — 스캔본·암호·손상 PDF는 열 수 없습니다.")
    else:
        chunks = chunk_text(raw_text, 500, 50)
        # [연결] 에이전트의 search_docs 도구가 같은 세션에서 이 청크를 검색한다
        #   (session 사이드채널 — classify_image와 동일 계열).
        st.session_state["rag_chunks"] = chunks
        st.session_state["rag_doc_name"] = uploaded.name
        st.caption(f"청크 {len(chunks)}개 생성 — 이제 🤖 에이전트 페이지에서도 `문서에서 … 찾아줘`가 동작합니다.")

        # [왜] 문서를 바꾸면 이전 문서의 답변·출처가 섞여 보인다 — 파일이 달라지면 대화를 새로 시작(교안과 동일 규약).
        doc_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("showcase_rag_doc") != doc_key:
            st.session_state.showcase_rag_doc = doc_key
            st.session_state.showcase_rag_msgs = []

        for m in st.session_state.showcase_rag_msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("context"):
                    with st.expander("📄 검색되어 전달된 후보 청크"):
                        st.markdown(m["context"])

        if prompt := st.chat_input("이 문서에 대해 궁금한 것을 물어보세요"):
            st.session_state.showcase_rag_msgs.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.status("문서에서 답을 찾는 중...", expanded=True) as status:
                    st.write("1) 관련 청크 검색 중...")
                    matches = retrieve(prompt, chunks, emb or "local")
                    st.write(f"2) 상위 {len(matches)}개 청크 → LLM 전달 중...")
                    context = assemble_context(matches)
                    answer = answer_with_context(prompt, context, llm or "local")
                    status.update(label="완료", state="complete", expanded=False)
                st.markdown(answer)
                with st.expander("📄 검색되어 전달된 후보 청크"):
                    st.markdown(context)
            st.session_state.showcase_rag_msgs.append({"role": "assistant", "content": answer, "context": context})
