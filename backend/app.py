"""
app.py — HuggingFace Spaces entry point (Gradio SDK)

This file allows the FastAPI backend to be hosted on HuggingFace Spaces
using the FREE Gradio SDK (no Docker, no paid plan required).

How it works:
  - Gradio Spaces expects a file called app.py with a `demo` Gradio object
  - Gradio runs on FastAPI internally
  - We use gr.mount_gradio_app() to expose our existing FastAPI routes
    alongside a minimal Gradio UI
  - Our full REST API is available at /api/analyze, /api/health, etc.
"""

import gradio as gr
from api.main import app as fastapi_app


# ── Minimal Gradio UI (required by HF Spaces) ─────────────────────────────────
with gr.Blocks(
    title="Clinical AI Copilot — API",
    theme=gr.themes.Soft(),
    css="""
        .container { max-width: 700px; margin: auto; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; 
                 background: #0d1528; color: #00d4ff; font-size: 0.85rem; }
    """
) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("""
        # 🧠 Clinical AI Copilot — Backend API
        
        **Multimodal AI system for chest X-ray analysis.**  
        Uses LangGraph orchestration, MCP tools, and Google Gemini 1.5 Flash.
        
        ---
        
        ## API Endpoints
        
        | Endpoint | Method | Description |
        |---|---|---|
        | `/api/health` | GET | Health check |
        | `/api/analyze` | POST | Analyze X-ray (SSE stream) |
        | `/api/report/{id}/pdf` | GET | Download PDF report |
        | `/docs` | GET | Swagger UI |
        
        ---
        
        > ⚠️ **Research & portfolio demonstration only. Not a medical device.**
        """)
        
        gr.HTML("""
        <div style="text-align:center; margin-top: 20px;">
            <a href="/docs" target="_blank" 
               style="background: #00d4ff; color: #000; padding: 10px 24px; 
                      border-radius: 8px; text-decoration: none; font-weight: 600;">
                📖 Open Swagger UI
            </a>
        </div>
        """)


# ── Mount the full FastAPI app onto Gradio ────────────────────────────────────
# This exposes all /api/* routes alongside the Gradio interface at /gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
