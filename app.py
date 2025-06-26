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
import io
from typing import Dict

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    redirect,
    url_for,
)
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------- Optional file‑parsing helpers -----------------------------------

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None

try:
    import docx  # python‑docx  # type: ignore
except ImportError:
    docx = None


#------configuration-------#

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var before starting the server.")

openai.api_key = OPENAI_API_KEY



# ---------- Utility functions -----------------------------------------------

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    if not pdfplumber:
        raise RuntimeError("pdfplumber not installed on server")
    text_chunks = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    if not docx:
        raise RuntimeError("python‑docx not installed on server")
    document = docx.Document(file_stream)
    return "\n".join([p.text for p in document.paragraphs])


def allowed_extension(filename: str) -> bool:
    return filename.lower().endswith((".pdf", ".docx"))


def read_resume_or_desc(field_name: str) -> str:
    """Return textual content for resume or job description.

    Priority: file upload → textarea / JSON input.
    """
    # 1️⃣ File upload (only for form/multipart requests):
    if field_name in request.files and request.files[field_name]:
        f = request.files[field_name]
        if f.filename and allowed_extension(f.filename):
            ext = os.path.splitext(f.filename)[1].lower()
            file_bytes = io.BytesIO(f.read())
            try:
                if ext == ".pdf":
                    return extract_text_from_pdf(file_bytes)
                elif ext == ".docx":
                    return extract_text_from_docx(file_bytes)
            except Exception as e:  # pragma: no cover
                # Return empty string so downstream validation fails cleanly.
                print("File‑parse error:", e)
                return ""
    # 2️⃣ Fallback to textarea (form) or JSON payload:
    if request.form:
        return request.form.get(field_name.replace("_file", ""), "").strip()
    #return (request.json or {}).get(field_name.replace("_file", ""), "").strip()
    val = (request.json or {}).get(field_name.replace("_file", ""))
    return val.strip() if isinstance(val, str) else ""


# ---------- Prompt builder ---------------------------------------------------

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

    return system_prompt, user_content

# ---------- Flask app --------------------------------------------------------

app = Flask(__name__)

HTML_FORM = """
<!doctype html>
<title>Why Was I Rejected?</title>
<h1>Why Was I Rejected?</h1>
<form method="post" action="/analyze" enctype="multipart/form-data" style="max-width:600px">
  <h3>Resume</h3>
  <input type="file" name="resume_file" accept=".pdf,.docx">
  <br><small>or paste below</small><br>
  <textarea name="resume" rows="8" cols="80"></textarea>
  <hr>
  <h3>Job Description</h3>
  <input type="file" name="job_desc_file" accept=".pdf,.docx">
  <br><small>or paste below</small><br>
  <textarea name="job_desc" rows="8" cols="80"></textarea>
  <br><br>
  <button type="submit">Analyze</button>
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
    resume_text = read_resume_or_desc("resume_file")
    job_desc_text = read_resume_or_desc("job_desc_file")

    if not resume_text or not job_desc_text:
        return (
            jsonify(
                {
                    "error": "Both resume and job description are required — either upload a PDF/DOCX or paste the text."
                }
            ),
            400,
        )

    system_prompt, user_prompt = build_prompt(resume_text, job_desc_text)

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

    # JSON API support remains intact
    if request.content_type and request.content_type.startswith("application/json"):
        return jsonify({"analysis": analysis})
    return render_template_string(HTML_FORM, result=analysis)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
