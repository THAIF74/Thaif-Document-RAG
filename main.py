import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Load NVIDIA API key from .env in this project folder.
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY was not found. Keep your existing .env file in the project folder."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    timeout=90.0,
    max_retries=2,
)

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
CHAT_MODEL = "openai/gpt-oss-20b"

chunks = []
vectors = None
full_document_text = ""
pdf_name = ""


def split_text(text, size=220, overlap=40):
    """Split document text into overlapping word chunks."""
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step)]


def get_embeddings(texts):
    """Create embeddings in small batches and keep the returned order stable."""
    all_vectors = []
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]
        result = client.embeddings.create(model=EMBED_MODEL, input=batch)
        ordered = sorted(result.data, key=lambda item: item.index)
        all_vectors.extend(item.embedding for item in ordered)
    return np.asarray(all_vectors, dtype=np.float32)


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def upload_pdf():
    global chunks, vectors, full_document_text, pdf_name

    path = filedialog.askopenfilename(
        title="Choose a PDF document",
        filetypes=[("PDF files", "*.pdf")],
    )
    if not path:
        return

    upload_button.config(state=tk.DISABLED)
    ask_button.config(state=tk.DISABLED)
    try:
        status_var.set("Reading PDF...")
        root.update_idletasks()

        reader = PdfReader(path)
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                page_texts.append(f"[PAGE {page_number}]\n{page_text}")

        full_text = "\n\n".join(page_texts).strip()
        if not full_text:
            raise ValueError(
                "No selectable text was found in this PDF. It may be scanned; OCR is required first."
            )

        new_chunks = split_text(full_text)
        if not new_chunks:
            raise ValueError("No text chunks could be created from this PDF.")

        status_var.set(f"Creating embeddings for {len(new_chunks)} chunks...")
        root.update_idletasks()
        new_vectors = get_embeddings(new_chunks)
        if len(new_vectors) != len(new_chunks):
            raise RuntimeError("Embedding count did not match the number of text chunks.")

        chunks = new_chunks
        vectors = new_vectors
        full_document_text = full_text
        pdf_name = os.path.basename(path)

        chat_box.insert(
            tk.END,
            f"\nLoaded PDF: {pdf_name}\nPages: {len(reader.pages)} | Text chunks: {len(chunks)}\n",
        )
        chat_box.see(tk.END)
        status_var.set(f"Ready. Loaded {pdf_name}")
    except Exception as error:
        status_var.set("PDF loading failed.")
        messagebox.showerror(
            "PDF Error",
            f"{type(error).__name__}: {error}\n\n"
            "Check your PDF, internet connection, NVIDIA API key, and model access.",
        )
    finally:
        upload_button.config(state=tk.NORMAL)
        ask_button.config(state=tk.NORMAL)


def get_answer_text(response):
    """Safely extract a text answer from OpenAI-compatible API responses."""
    if not getattr(response, "choices", None):
        return "The API returned no choices. Check NVIDIA model availability and try again."

    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return "The API returned no message. Check NVIDIA model availability and try again."

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some compatible endpoints expose text in alternate fields.
    for field in ("output_text", "reasoning_content"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    finish_reason = getattr(choice, "finish_reason", None) or "not provided"
    return (
        "The model API returned an empty text answer (finish reason: "
        f"{finish_reason}). This is an API/model response issue, not a PDF answer. "
        "Please try again; if it repeats, check NVIDIA model availability."
    )


def ask_question():
    question = question_entry.get().strip()

    if not question:
        messagebox.showinfo("Question needed", "Type a question first.")
        return

    if not chunks or vectors is None or len(vectors) == 0:
        messagebox.showinfo(
            "Upload a PDF",
            "Please upload a PDF before asking questions."
        )
        return

    ask_button.config(state=tk.DISABLED)

    try:
        status_var.set("Searching document...")
        root.update_idletasks()

        lower_question = question.lower()

        # Detect questions that need information from the whole PDF.
        wants_all = any(term in lower_question for term in (
            "all questions",
            "all the questions",
            "list all",
            "complete document",
            "entire document",
            "full document",
            "marks allocated",
            "marks distribution",
            "total marks",
            "each question",
            "all marks",
            "part a",
            "part b",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "exactly as extracted",
        ))

        if wants_all:
            context = full_document_text.strip()

            if not context:
                context = "\n\n".join(chunks)
        else:
            query_vector = get_embeddings([question])[0]
            scores = [
                cosine_similarity(query_vector, vector)
                for vector in vectors
            ]

            number_to_use = min(6, len(chunks))
            top_indices = np.argsort(scores)[-number_to_use:][::-1]

            context = "\n\n".join(
                chunks[int(i)] for i in top_indices
            )

        status_var.set("Generating answer...")
        root.update_idletasks()

        system_prompt = """
You are a PDF question-answering assistant.

RULES:
1. Answer the user's request directly.
2. Never describe your reasoning or explain what you plan to do.
3. Use only the supplied PDF text.
4. Never guess, invent, or calculate missing question marks.
5. Preserve question numbers, subquestion letters, question wording,
   and marks as they appear in the supplied text.
6. If a question or its marks cannot be read from the text, write
   "Not clearly available in extracted text".
7. When asked to list questions and marks, output a Markdown table
   with these columns:
   Section | Question Number | Question | Marks
8. Include every identifiable requested question, including OR options.
9. Do not omit questions just to shorten the answer.
10. Output only the requested answer, without an introduction,
    explanation, or conclusion.
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "PDF EXTRACTED TEXT:\n"
                        "--------------------\n"
                        f"{context}\n"
                        "--------------------\n\n"
                        "TASK:\n"
                        f"{question}\n\n"
                        "Return the requested result directly."
                    ),
                },
            ],
            temperature=0,
            max_tokens=5000,
        )

        answer = get_answer_text(response)

        if not answer or not answer.strip():
            answer = (
                "The AI returned an empty answer. "
                "Please try again."
            )

        chat_box.insert(
            tk.END,
            f"\nYou: {question}\n\nAI: {answer}\n"
        )
        chat_box.see(tk.END)

        question_entry.delete(0, tk.END)
        status_var.set(f"Answered using {pdf_name}")

    except Exception as error:
        status_var.set("Request failed.")
        messagebox.showerror(
            "AI Error",
            f"{type(error).__name__}: {error}\n\n"
            "Check your internet connection, NVIDIA API key, "
            "and model access.",
        )

    finally:
        ask_button.config(state=tk.NORMAL)


root = tk.Tk()
root.title("Thaif Document RAG")
root.geometry("800x620")
root.minsize(600, 450)

title_label = tk.Label(root, text="Thaif Document RAG", font=("Arial", 20, "bold"))
title_label.pack(pady=(14, 4))

subtitle = tk.Label(
    root,
    text="Upload a PDF and ask questions about its content",
    font=("Arial", 10),
)
subtitle.pack(pady=(0, 10))

upload_button = tk.Button(root, text="Upload PDF", command=upload_pdf, width=18)
upload_button.pack(pady=5)

status_var = tk.StringVar(value="Please upload a PDF.")
status_label = tk.Label(root, textvariable=status_var, fg="blue", wraplength=740)
status_label.pack(pady=5)

chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 10), state=tk.NORMAL)
chat_box.pack(padx=14, pady=10, fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, padx=14, pady=(0, 14))

question_entry = tk.Entry(bottom_frame, font=("Arial", 11))
question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7, padx=(0, 8))

ask_button = tk.Button(bottom_frame, text="Ask AI", command=ask_question, width=12)
ask_button.pack(side=tk.RIGHT)
question_entry.bind("<Return>", lambda event: ask_question())

root.mainloop()
