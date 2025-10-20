"""FastAPI application exposing a simple HTML form to run the Zhiwei RAG workflow."""

from __future__ import annotations

import html
from typing import Optional

import logging

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import os
from qdrant_workflow import LLM_MODEL, run_qdrant_rag_workflow

logger = logging.getLogger(__name__)

app = FastAPI(title="Zhiwei Doushu RAG Assistant")

LLM_MODEL_OPTIONS = [
    ("gpt-4o", "gpt-4o"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-4.1-mini", "gpt-4.1-mini"),
]

BIRTH_HOUR_OPTIONS = [
    ('0',"0:00-0:59"),
    ('1',"1:00-2:59"),
    ('2',"3:00-4:59"),
    ('3',"5:00-6:59"),
    ('4',"7:00-8:59"),
    ('5',"9:00-10:59"),
    ('6',"11:00-12:59"),
    ('7',"13:00-14:59"),
    ('8',"15:00-16:59"),
    ('9',"17:00-18:59"),
    ('10',"19:00-20:59"),
    ('11',"21:00-22:59"),
    ('12',"23:00-23:59")
    ]


def _render_form(
    *,
    birth_date: str = "",
    birth_hour: str = "",
    gender: str = "",
    question: str = "",
    llm_model: str = "",
    api_key: str = "",
    answer: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    """Render the HTML page with the input form and optional results."""

    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    llm_value = llm_model or ""
    llm_options_html = "\n".join(
        f"<option value=\"{esc(value)}\" {'selected' if llm_value == value else ''}>{esc(label)}</option>"
        for value, label in LLM_MODEL_OPTIONS
    )

    birth_hour_value = birth_hour or ""
    birth_hour_options_html = "\n".join(
        f"<option value=\"{esc(value)}\" {'selected' if birth_hour_value == value else ''}>{esc(label)}</option>"
        for value, label in BIRTH_HOUR_OPTIONS
    )

    answer_block = ""
    if answer:
        answer_block = f"""
        <section class=\"result\">
          <h2>ANSWER</h2>
          <pre>{esc(answer)}</pre>
        </section>
        """
    elif error:
        answer_block = f"""
        <section class=\"result error\">
          <h2>ERROR</h2>
          <pre>{esc(error)}</pre>
        </section>
        """

    return f"""
    <!DOCTYPE html>
    <html lang=\"zh\">
    <head>
      <meta charset=\"utf-8\" />
      <title>紫微斗数 RAG 试算</title>
      <style>
        body {{
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 0 auto;
          padding: 2rem 1.5rem 3rem;
          max-width: 720px;
          background: #f4f6fb;
          color: #1f2430;
        }}
        header {{
          margin-bottom: 1.5rem;
          text-align: center;
        }}
        form {{
          background: #ffffff;
          padding: 1.5rem;
          border-radius: 12px;
          box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
          display: grid;
          gap: 1rem;
        }}
        label {{
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
          font-weight: 600;
        }}
        input, select, textarea {{
          border: 1px solid #d0d7e2;
          border-radius: 8px;
          padding: 0.6rem 0.75rem;
          font-size: 1rem;
          font-family: inherit;
          resize: vertical;
        }}
        button {{
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          color: #ffffff;
          border: none;
          border-radius: 999px;
          padding: 0.75rem 1.75rem;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.15s ease, box-shadow 0.15s ease;
          justify-self: start;
        }}
        button:hover {{
          transform: translateY(-1px);
          box-shadow: 0 10px 22px rgba(99, 102, 241, 0.25);
        }}
        button:active {{
          transform: translateY(0);
        }}
        button:disabled {{
          opacity: 0.6;
          cursor: not-allowed;
        }}
        .helper {{
          font-weight: 400;
          font-size: 0.9rem;
          color: #5b6475;
        }}
        .result {{
          margin-top: 1.75rem;
          background: #ffffff;
          border-radius: 12px;
          padding: 1.5rem;
          box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }}
        .result h2 {{
          margin-top: 0;
          font-size: 1.25rem;
        }}
        .result pre {{
          white-space: pre-wrap;
          word-break: break-word;
          font-family: "Fira Code", "Consolas", "SFMono-Regular", monospace;
          line-height: 1.6;
        }}
        .result.error {{
          border-left: 4px solid #f97316;
        }}
        .loading-overlay {{
          position: fixed;
          inset: 0;
          background: rgba(15, 23, 42, 0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 10;
          backdrop-filter: blur(2px);
        }}
        .loading-overlay.hidden {{
          display: none;
        }}
        .loading-card {{
          background: #ffffff;
          border-radius: 12px;
          padding: 1.75rem 2.25rem;
          box-shadow: 0 16px 36px rgba(15, 23, 42, 0.22);
          text-align: center;
        }}
        .spinner {{
          width: 34px;
          height: 34px;
          border-radius: 50%;
          border: 4px solid #e0e7ff;
          border-top-color: #6366f1;
          margin: 0 auto 1rem;
          animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
          to {{
            transform: rotate(360deg);
          }}
        }}
      </style>
    </head>
    <body>
      <div id=\"loading-overlay\" class=\"loading-overlay hidden\">
        <div class=\"loading-card\">
          <div class=\"spinner\"></div>
          <p>Loading... Please wait...</p>
        </div>
      </div>
      <header>
        <h1>紫微斗数 RAG Assistant</h1>
        <p>Enter your birth details and question; the LLM will provide you with professional analysis.</p>
      </header>
      <form method=\"post\" action=\"/\" autocomplete=\"off\" id=\"rag-form\">
        <label>
          OpenAI API Key
          <input type=\"password\" name=\"api_key\" placeholder=\"sk-...\" autocomplete=\"off\" value=\"{esc(api_key)}\" />
        </label>
        <label>
          LLM model
          <select name=\"llm_model\">
            {llm_options_html}
          </select>
        </label>
        <label>
          Birth Date
          <input type=\"date\" name=\"birth_date\" required value=\"{esc(birth_date)}\" />
        </label>
        <label>
          Time of birth (24-hour format)
          <select name=\"birth_hour\" required>
            {birth_hour_options_html}
          </select>
        </label>
        <label>
          Gender
          <select name=\"gender\" required>
            <option value=\"男\" {"selected" if gender in {"男", "Male"} else ""}>男 (Male)</option>
            <option value=\"女\" {"selected" if gender in {"女", "Female"} else ""}>女 (Female)</option>
          </select>
        </label>
        <label>
          Question:
          <textarea name=\"question\" rows=\"5\" placeholder=\"Please describe the life question you'd like to ask about...\" required>{esc(question)}</textarea>
        </label>
        <button type=\"submit\" id=\"submit-button\">RUN</button>
      </form>
      {answer_block}
      <script>
        const form = document.getElementById('rag-form');
        const overlay = document.getElementById('loading-overlay');
        const submitButton = document.getElementById('submit-button');
        if (form && overlay && submitButton) {{
          form.addEventListener('submit', () => {{
            overlay.classList.remove('hidden');
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing…...';
          }});
        }}
      </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def load_form() -> HTMLResponse:
    """Render the initial form page."""
    return HTMLResponse(_render_form())


@app.post("/", response_class=HTMLResponse)
async def submit_form(
    birth_date: str = Form(...),
    birth_hour: int = Form(...),
    gender: str = Form(...),
    question: str = Form(...),
    api_key: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
) -> HTMLResponse:
    """Handle form submission and run the RAG workflow."""
    try:
        normalized_gender = {"Male": "男", "Female": "女"}.get(gender, gender)
        result = run_qdrant_rag_workflow(
            question=question,
            birth_date=birth_date,
            birth_hour=birth_hour,
            gender=normalized_gender,
            api_key=api_key or None,
            llm_model=llm_model or None,
        )
        answer = result.get("answer") or "ERROR"
        html_content = _render_form(
            birth_date=birth_date,
            birth_hour=str(birth_hour),
            gender=normalized_gender,
            question=question,
            llm_model=llm_model or "",
            api_key="",
            answer=answer,
        )
        return HTMLResponse(html_content)
    except Exception as exc:  # pragma: no cover - surface error to UI
        logger.exception("RAG workflow invocation failed")
        html_content = _render_form(
            birth_date=birth_date,
            birth_hour=str(birth_hour),
            gender=normalized_gender,
            question=question,
            llm_model=llm_model or "",
            api_key="",
            error=str(exc),
        )
        return HTMLResponse(html_content, status_code=500)
