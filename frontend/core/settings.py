from __future__ import annotations

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from core.constants import DEFAULT_BATCH_URL, DEFAULT_PREDICT_URL


def _get_secret_or_env(name: str, default: str) -> str:
    env_value = os.getenv(name, default)
    try:
        return str(st.secrets.get(name, env_value))
    except StreamlitSecretNotFoundError:
        return env_value


def get_predict_url() -> str:
    return _get_secret_or_env("API_URL_PREDICT", DEFAULT_PREDICT_URL)


def get_batch_url() -> str:
    return _get_secret_or_env("API_URL_BATCH", DEFAULT_BATCH_URL)
