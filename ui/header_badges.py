"""Shared header badge links (shields.io) for Paper and GitHub."""

from __future__ import annotations

import streamlit as st

_PAPER_BADGE = (
    "https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white"
)
_GITHUB_BADGE = (
    "https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white"
)


def render_header_badges(paper_url: str, github_url: str) -> None:
    """Render Paper and GitHub shields.io badge links."""
    col1, col2, _ = st.columns([1, 2, 8])
    with col1:
        st.markdown(f"[![Paper]({_PAPER_BADGE})]({paper_url})")
    with col2:
        st.markdown(f"[![GitHub]({_GITHUB_BADGE})]({github_url})")
