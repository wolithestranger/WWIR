"""
Flask backend for the **Why Was I Rejected?** resume analyzer.
Minimal MVP: accepts raw resume & job‑description text as JSON, sends them to
OpenAI GPT, and returns the analysis in JSON.

Run locally:
$ export OPENAI_API_KEY="sk‑..."
$ pip install -r requirements.txt
$ python app.py

Then test with:
$ curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume": "...", "job_description": "..."}'
"""

import os
from flask import Flask, request, jsonify
import openai
from typing import Dict


#------configuration-------#

OPENAI_API_KEY = os.getenv("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")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var before starting the server.")

openai.api_key = OPENAI_API_KEY

def build_prompt( resume: str, job_description: str)-> str:
    """Return the system/user prompt for GPT."""

    system_prompt = (
        "You are an expert technical career coach. "
        "Compare the candidate's resume to the position description. "
        "Identify skill gaps, missing keywords, and concrete improvements. "
        "Output structured advice in four sections:\n"
        "1. Overall Fit Summary (2‑3 sentences)\n"
        "2. Missing or Weak Keywords/Skills (bullet list)\n"
        "3. Resume Improvement Suggestions (bullet list)\n"
        "4. General Job‑Search Advice (≤150 words)"
    )

    user_content = (
        f"[RESUME]\n{resume}\n\n[JOB_DESCRIPTION]\n{job_description}"
    )

#------flask application-----#

app =  Flask (__name__)

@app.route("/health", methods=["GET"])
def health()-> Dict[str,str]:
    """Simple liveness probe."""
    return {"status": "ok"}

@app.route("/analyze", methods=["POST"])
def analyze():
    """Main endpoint. Expects JSON with 'resume' and 'job_description'."""

    data = request.get_json(silent=True) or {}
    resume = data.get("resume", "").strip()
    job_desc = data.get("job_description", "").strip()

    if not resume or not job_desc:
        return jsonify({"error": "'resume' and 'job_description' are required."}), 400
    
    system_prompt, user_prompt = build_prompt(resume, job_desc)

    try:
        response = openai.ChatCompletion.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
    except openai.error.OpenAIError as e:
        return jsonify({"error": str(e)}), 500
    
    analysis = response.choices[0].message.content.strip()
    return jsonify({"analysis": analysis})

if __name__ == "__main__":
    # Allow port override for deployment platforms (Render, Railway, etc.)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)