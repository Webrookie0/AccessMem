from pathlib import Path
from urllib.parse import urlsplit

import requests
from streamlit.testing.v1 import AppTest


def test_streamlit_login_chat_and_logout_use_accessmem_api(
    client, seed_store, stub_model, monkeypatch
):
    def post(url, *, data=None, json=None, headers=None, timeout=None):
        return client.post(urlsplit(url).path, data=data, json=json, headers=headers)

    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"))
    app.run(timeout=10)
    assert not app.exception and "AccessMem" in app.title[0].value
    app.text_input[0].input("bob")
    app.text_input[1].input("demo")
    app.button[0].click().run(timeout=10)
    assert not app.exception and app.session_state["user"]["role"] == "manager"
    app.chat_input[0].set_value("What is the Q3 forecast?").run(timeout=10)
    assert not app.exception
    assert any("$148.3M" in markdown.value for markdown in app.markdown)
    assert any("RRF score" in caption.value for caption in app.caption)
    app.button[0].click().run(timeout=10)
    assert app.session_state["user"] is None and app.session_state["history"] == []


def test_streamlit_handles_failed_authentication(client, monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: client.post(urlsplit(url).path, data=kwargs["data"]),
    )
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py")
    ).run()
    app.text_input[1].input("wrong")
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Invalid username or password."
