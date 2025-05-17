import fitz  # PyMuPDF
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

# === Together AI Setup ===
TOGETHER_API_KEY = "1af14711e4bd43a557c077dd6c09c21b88f25c5142b16965173f605b61449333"
TOGETHER_MODEL = "togethercomputer/llama-2-7b-chat"  # ✅ Free model

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

    extracted_text = extracted_text[:4000]  # Truncate to fit model

    # === Prepare Prompt ===
    base_prompt = (
        "Summarize the following into clear bullet points:\n"
        "- Highlight key ideas and tasks\n"
        "- Use simple language\n"
        "- Include important dates or names if mentioned\n\n"
    )

    final_prompt = f"{prompt.strip()}\n\n{extracted_text}" if prompt else base_prompt + extracted_text
    print("finalprompt", final_prompt)
    # === Call Together AI ===
    summary = await query_together_ai(final_prompt)
    print("summary", summary)

    # === Generate and Return PDF ===
    pdf_file = generate_pdf(summary)
    return StreamingResponse(BytesIO(pdf_file), media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=summary.pdf"
    })


# === Text Extraction Helpers ===

def extract_text_from_pdf(data):
    text = ""
    doc = fitz.open(stream=data, filetype="pdf")
    for page in doc:
        text += page.get_text()
        print(text.strip())
    return text.strip()

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

    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.together.xyz/inference", json=payload, headers=headers)
        result = response.json()
        return result.get("output", "AI failed to summarize.").strip()


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
