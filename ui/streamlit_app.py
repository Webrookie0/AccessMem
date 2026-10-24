"""AccessMem chat interface."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "200"))
DEMO = os.getenv("DEMO_USERS_ENABLED", "true").lower() == "true"

st.set_page_config(page_title="AccessMem", page_icon="🔐", layout="centered")
st.title("🔐 AccessMem")
st.caption("Ask questions about documents available to your tenant and clearance.")

for key, value in (("token", None), ("user", None), ("history", [])):
    if key not in st.session_state:
        st.session_state[key] = value


def login(username: str, password: str) -> str | None:
    try:
        response = requests.post(
            f"{API}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
    except requests.RequestException:
        return "Cannot reach AccessMem. Start the API and try again."
    if response.status_code == 401:
        return "Invalid username or password."
    if response.status_code == 429:
        return "Too many sign-in attempts. Please try again later."
    if response.status_code != 200:
        return "Sign-in failed. Please try again."
    try:
        body = response.json()
        st.session_state.token = body["access_token"]
        st.session_state.user = body["user"]
    except (ValueError, KeyError):
        return "The API returned an invalid sign-in response."
    st.session_state.history = []
    return None


def logout() -> None:
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.history = []


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**{source['source']}**")
            st.caption(f"Clearance: {source['clearance']} · Tenant: {source['tenant_id']}")
            if source.get("score") is not None:
                st.caption(f"Cosine similarity: {source['score']:.4f}")
            if source.get("rrf_score") is not None:
                st.caption(f"RRF score: {source['rrf_score']:.6f}")


def show_answer(message: dict) -> None:
    if message.get("blocked"):
        st.warning(message["content"])
    elif message.get("refused"):
        st.info(message["content"])
    else:
        st.markdown(message["content"])
    show_sources(message.get("sources", []))


with st.sidebar:
    st.header("Sign in")
    if st.session_state.user:
        user = st.session_state.user
        st.success(user["username"])
        st.write(f"Role: {user['role']}")
        st.write(f"Tenant: {user['tenant_id']}")
        st.write(f"Clearance: {user['clearance']}")
        if st.button("Log out"):
            logout()
            st.rerun()
    else:
        with st.form("login_form"):
            username = st.text_input("Username", value="alice" if DEMO else "")
            password = st.text_input("Password", value="demo" if DEMO else "", type="password")
            if st.form_submit_button("Log in"):
                error = login(username.strip(), password)
                if error:
                    st.error(error)
                else:
                    st.rerun()
        if DEMO:
            with st.expander("Demo accounts"):
                st.write("alice: junior at Acme")
                st.write("bob: manager at Acme")
                st.write("carol: C-suite at Acme")
                st.write("dave: C-suite at Globex")
                st.caption("Demo password: demo")

    st.divider()
    search_label = st.selectbox(
        "Search mode",
        ["Hybrid", "Semantic"],
        help="Hybrid combines keyword matches and meaning. Semantic searches by meaning.",
    )
    retrieval_mode = "hybrid" if search_label == "Hybrid" else "dense"
    st.markdown(
        "**Try these prompts**\n\n"
        "- What is Acme's vacation policy?\n"
        "- What is the Q3 forecast?\n"
        "- Tell me about Project Northwind.\n"
        "- What is Globex's poison pill plan?"
    )
    if st.button("Check service"):
        try:
            response = requests.get(f"{API}/ready", timeout=10)
            if response.status_code == 200:
                st.success("AccessMem is ready.")
            else:
                st.warning("Check Ollama, configured models, and document ingestion.")
        except requests.RequestException:
            st.error("Cannot reach the API.")

if not st.session_state.user:
    st.info("Sign in from the sidebar to start chatting.")
    st.stop()

st.caption("Each question is answered independently. Chat history stays in this session.")
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message)
        else:
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents"):
            try:
                response = requests.post(
                    f"{API}/chat",
                    json={"question": prompt, "retrieval_mode": retrieval_mode},
                    headers={"Authorization": f"Bearer {st.session_state.token}"},
                    timeout=API_TIMEOUT,
                )
            except requests.RequestException:
                st.error("Cannot reach AccessMem. Please try again.")
                st.stop()
        if response.status_code == 401:
            logout()
            st.error("Session expired. Sign in again.")
            st.stop()
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "The request failed.")
            except ValueError:
                detail = "The request failed."
            st.error(detail if isinstance(detail, str) else "The request failed.")
            st.stop()
        try:
            data = response.json()
            message = {
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
                "blocked": data.get("blocked", False),
                "refused": data.get("refused", False),
            }
        except (ValueError, KeyError):
            st.error("The API returned an invalid answer.")
            st.stop()
        show_answer(message)
        st.session_state.history.append(message)
