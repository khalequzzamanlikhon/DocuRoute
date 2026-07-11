

"""
Minimal Streamlit demo UI. Run with:
    streamlit run frontend/app.py

Talks to the FastAPI backend over HTTP so the demo reflects exactly what
the deployed API returns (same trace, same citations).
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

# Updated page title to match the project name
st.set_page_config(page_title="DocuRoute", page_icon="🔎")

# New Title
st.title("🔎 DocuRoute")

# Brief description explaining the "Agentic RAG" concept 
st.markdown(
    "**An intelligent, query-routing RAG system.** DocuRoute dynamically routes your questions "
    "to the right tool—extracting either narrative insights or structured numerical data on the fly."
)

# Context about the specific documents
st.caption(
    "Currently querying ingested financial reports: `etsy-20231231.pdf`, `etsy-20241231.pdf`, and `etsy-20251231.pdf`."
)

with st.expander("💡 Try these sample questions:"):
    st.markdown(
        """
        - *Which year had the highest Income from operations: 2023, 2024, or 2025?*
        - *What is the Initial Conversion Price and Maturity Date for Etsy's convertible notes?*
        - *Explain the "internal carbon pricing framework" implemented in 2025 and how it informs the company's "emissions budget.*
        - *How did Etsy support its sellers impacted by the 2025 Los Angeles fires and other FEMA-declared disasters?*
        """
    )

question = st.text_input(
    "Your question", 
    placeholder="e.g. What was Etsy's total revenue in 2023?"
)

if st.button("Ask") and question.strip():
    with st.spinner("Routing, retrieving, and generating..."):
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}. Is the API running (`uvicorn api.main:app`)?")
            st.stop()

    st.subheader("Answer")
    if data.get("refused"):
        st.warning(data["answer"])
    else:
        st.write(data["answer"])

    st.subheader("Routing")
    st.write(f"**Route:** `{data['route']}` — {data['route_reasoning']}")
    if data.get("generated_sql"):
        st.code(data["generated_sql"], language="sql")

    if data.get("citations"):
        st.subheader("Citations")
        for c in data["citations"]:
            if c["kind"] == "chunk":
                st.markdown(f"- **[{c['marker']}]** {c['doc_id']}, page {c['page_number']}")
            else:
                st.markdown(f"- **[{c['marker']}]** SQL: `{c['sql']}`")