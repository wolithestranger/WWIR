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
from flask import Flask, request, jsonify, render_template_string
import openai
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


#------configuration-------#

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

app = Flask(__name__)

HTML_FORM = """
<!doctype html>
<title>Why Was I Rejected?</title>
<h1>Why Was I Rejected?</h1>
<form method=post action="/analyze" style="max-width:600px">
  <label>Paste your Resume:</label><br>
  <textarea name=resume rows=10 cols=80 required></textarea><br><br>
  <label>Paste the Job Description:</label><br>
  <textarea name=job_description rows=10 cols=80 required></textarea><br><br>
  <button type=submit>Analyze</button>
</form>
{% if result %}
<hr>
<h2>Analysis</h2>
<pre>{{ result }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_FORM, result=None)

@app.route("/health", methods=["GET"])
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.route("/analyze", methods=["POST"])
def analyze():
    resume = request.form.get("resume") or request.json.get("resume", "")
    job_desc = request.form.get("job_description") or request.json.get("job_description", "")

    if not resume or not job_desc:
        return jsonify({"error": "'resume' and 'job_description' are required."}), 400

    system_prompt, user_prompt = build_prompt(resume, job_desc)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
    except openai.error.OpenAIError as e:
        return jsonify({"error": str(e)}), 500

    analysis = response.choices[0].message.content.strip()

    # If it's a form submission, render HTML; else return JSON
    if request.content_type.startswith("application/json"):
        return jsonify({"analysis": analysis})
    else:
        return render_template_string(HTML_FORM, result=analysis)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)