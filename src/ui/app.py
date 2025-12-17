"""Minimal FastAPI UI with pale blue background."""

from __future__ import annotations

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from src.agents.cognitive_agent import build_demo_agent

app = FastAPI(title="CogEval UI")
agent = build_demo_agent()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = """
    <html>
      <head>
        <style>
          body { background: #e6f1fb; font-family: Arial, sans-serif; }
          .container { max-width: 640px; margin: 2rem auto; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
          textarea { width: 100%; height: 120px; }
          button { background: #2a6fdb; color: white; padding: 0.5rem 1rem; border: none; border-radius: 6px; }
          .log { margin-top: 1rem; }
          .progress { color: #205ea8; margin-bottom: 0.5rem; }
          .controls { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
          .controls button.disabled { background: #c7d9f4; color: #6c7a92; cursor: not-allowed; }
        </style>
      </head>
      <body>
        <div class="container">
          <h2>CogEval Chat (Demo)</h2>
          <div class="progress">Progress: Intake → Tasks → Wrap-up</div>
          <form method="post" action="/chat">
            <textarea name="message" placeholder="Type here..."></textarea>
            <div class="controls">
              <button type="submit">Send</button>
              <button type="button" class="disabled">🎤 Voice (coming soon)</button>
              <button type="button" class="disabled">🎥 Video (coming soon)</button>
            </div>
          </form>
          <div class="log">
            <p>This demo performs text-only dialogue and uses placeholder speech/video controls.</p>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/chat")
def chat(message: str = Form(...)) -> str:
    return agent.run_turn(user_id="demo-user", session_id="demo-session", message=message)


__all__ = ["app"]
