import pytesseract
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from docx import Document
from pdfminer.high_level import extract_text  
import re

# === Together AI Setup ===
TOGETHER_API_KEY = "1af14711e4bd43a557c077dd6c09c21b88f25c5142b16965173f605b61449333"
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1" 

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize_note")
async def summarize_note(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    file_type = file.content_type
    raw_data = await file.read()

    # === Extract Text ===
    if file_type == "application/pdf":
        extracted_text = extract_text_from_pdf(raw_data)
    elif file_type in ["image/png", "image/jpeg"]:
        extracted_text = extract_text_from_image(raw_data)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(raw_data)
    else:
        return {"error": f"Unsupported file type: {file_type}"}

    extracted_text = extracted_text[:1500]  # Truncate for model reliability
    print("Extracted text:", extracted_text)

    # === RAG: Extract key facts for context ===
    rag_context = extract_key_facts(extracted_text)
    print("RAG context:", rag_context)

    # === Prepare Prompt ===
    base_prompt = (
        "You are an expert resume summarizer. Use only the provided context and text to generate a summary.\n"
        "Context (facts to use):\n"
        f"{rag_context}\n\n"
        "Summarize the following resume in 5-10 professional, fluent English bullet points:\n"
        "- Focus on key skills, education, and experience\n"
        "- Use clear, concise, and grammatically correct language\n"
        "- Do not copy raw text; rephrase in your own words\n"
        "- Include important projects and achievements if present\n\n"
    )

    if prompt and prompt.strip() and len(prompt.strip()) > 10:
        final_prompt = f"{prompt.strip()}\n\n{base_prompt}{extracted_text}"
    else:
        final_prompt = base_prompt + extracted_text

    print("Final prompt:", final_prompt)

    # === Try Together AI ===
    summary = await query_together_ai(final_prompt)

    # If Together AI fails, try web-based fallback
    if not summary.strip() or summary.strip() == "1/1":
        print("Together AI failed, trying web fallback...")
        summary = await fallback_web_summary(extracted_text, rag_context)
        if not summary.strip():
            summary = "No summary could be generated from the input."

    pdf_file = generate_pdf(summary)
    return StreamingResponse(BytesIO(pdf_file), media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=summary.pdf"
    })


# === Text Extraction Helpers ===

def extract_text_from_pdf(data):
    return extract_text(BytesIO(data)).strip()

def extract_text_from_image(data):
    image = Image.open(BytesIO(data))
    return pytesseract.image_to_string(image)

def extract_text_from_docx(data):
    file_stream = BytesIO(data)
    doc = Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])


# === AI Query ===

async def query_together_ai(prompt_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt_text,
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.together.xyz/inference",
                json=payload,
                headers=headers,
                timeout=20  # 20 seconds timeout
            )
            print("Together AI response:", response.text)
            result = response.json()
            try:
                return result["choices"][0]["text"].strip()
            except Exception:
                return "AI failed to summarize."
    except httpx.ReadTimeout:
        print("Together AI request timed out.")
        return ""
    except Exception as e:
        print("Together AI error:", e)
        return ""


async def fallback_web_summary(text: str, rag_context: str) -> str:
    """
    Fallback: Use a public summarization API if Together AI fails.
    Uses huggingface.co's inference API for facebook/bart-large-cnn (better English and summary quality).
    """
    import json
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Context (facts to use):\n{rag_context}\n\n"
        f"Summarize in professional, fluent English: {text[:1000]}"
    )
    payload = {"inputs": prompt}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, content=json.dumps(payload), timeout=30)
            print("HuggingFace response:", response.text)
            result = response.json()
            if isinstance(result, list) and result and "summary_text" in result[0]:
                return result[0]["summary_text"].strip()
            elif isinstance(result, dict) and "error" in result:
                return "[HuggingFace Error] " + result["error"]
    except Exception as e:
        print("Web fallback error:", e)
    return ""


# === PDF Generation ===

def generate_pdf(text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    for line in text.split('\n'):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line.strip())
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer.read()


def extract_key_facts(text: str) -> str:
    """
    Extracts key facts (emails, phone numbers, URLs, section headers) from the text for RAG context.
    """
    lines = text.splitlines()
    facts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r"@|\b\d{10}\b|https?://|www\\.", line, re.I):
            facts.append(line)
        elif line.lower() in ["education", "work experience", "skills", "projects"]:
            facts.append(f"Section: {line}")
    return "\n".join(facts[:10])
