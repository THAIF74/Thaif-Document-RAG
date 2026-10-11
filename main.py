
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY not found. Add it to your .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
CHAT_MODEL = "openai/gpt-oss-20b"

chunks = []
vectors = []
pdf_name = ""


def split_text(text, size=220, overlap=40):
    words = text.split()
    result = []
    start = 0

    while start < len(words):
        part = words[start:start + size]

        if part:
            result.append(" ".join(part))

        start += size - overlap

    return result


def get_embeddings(texts):
    result = []

    # Send smaller batches to the embedding API
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        ordered = sorted(
            response.data,
            key=lambda item: item.index
        )

        result.extend(
            [item.embedding for item in ordered]
        )

    return np.array(result, dtype=np.float32)


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def upload_pdf():
    global chunks, vectors, pdf_name

    path = filedialog.askopenfilename(
        title="Choose a PDF document",
        filetypes=[("PDF files", "*.pdf")]
    )

    if not path:
        return

    try:
        status_var.set("Reading PDF...")
        root.update_idletasks()

        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        text = "\n".join(pages).strip()

        if not text:
            messagebox.showwarning(
                "No text found",
                "This PDF may be scanned. Try a text-based PDF."
            )
            status_var.set("No readable text found.")
            return

        new_chunks = split_text(text)

        status_var.set(
            f"Creating embeddings for {len(new_chunks)} sections..."
        )
        root.update_idletasks()

        new_vectors = get_embeddings(new_chunks)

        chunks = new_chunks
        vectors = new_vectors
        pdf_name = os.path.basename(path)

        status_var.set(
            f"Ready: {pdf_name} ({len(chunks)} sections)"
        )

        chat_box.insert(
            tk.END,
            f"\nSystem: Loaded {pdf_name}. Ask me a question!\n"
        )
        chat_box.see(tk.END)

    except Exception as error:
        status_var.set("Could not load PDF.")
        messagebox.showerror("PDF Error", str(error))


def ask_question():
    question = question_entry.get().strip()

    if not question:
        messagebox.showinfo(
            "Enter a question",
            "Type a question about your PDF first."
        )
        return

    if not chunks:
        messagebox.showinfo(
            "Upload a PDF",
            "Upload a PDF before asking questions."
        )
        return

    ask_button.config(state=tk.DISABLED)

    try:
        status_var.set("Searching document...")
        root.update_idletasks()

        query_vector = get_embeddings([question])[0]

        scores = [
            cosine_similarity(query_vector, vector)
            for vector in vectors
        ]

        top_indices = np.argsort(scores)[-4:][::-1]

        context = "\n\n".join(
            chunks[i] for i in top_indices
        )

        status_var.set("Generating answer...")
        root.update_idletasks()

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using the "
                        "provided document excerpts. If the answer "
                        "is not supported by the excerpts, say that "
                        "the document does not provide enough "
                        "information. Do not invent facts."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT EXCERPTS:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    )
                }
            ],
            temperature=0.2,
            max_tokens=700
        )

        answer = response.choices[0].message.content

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
            f"{error}\n\nCheck your API key, internet, "
            "and model access."
        )

    finally:
        ask_button.config(state=tk.NORMAL)


root = tk.Tk()
root.title("Thaif Document RAG")
root.geometry("800x620")
root.minsize(600, 450)

title_label = tk.Label(
    root,
    text="Thaif Document RAG",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=(14, 4))

subtitle = tk.Label(
    root,
    text="Upload a PDF and ask questions about its content",
    font=("Arial", 10)
)
subtitle.pack(pady=(0, 10))

upload_button = tk.Button(
    root,
    text="Upload PDF",
    command=upload_pdf,
    width=18
)
upload_button.pack(pady=5)

status_var = tk.StringVar(value="Please upload a PDF.")
status_label = tk.Label(
    root,
    textvariable=status_var,
    fg="blue",
    wraplength=740
)
status_label.pack(pady=5)

chat_box = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 10),
    state=tk.NORMAL
)
chat_box.pack(
    padx=14,
    pady=10,
    fill=tk.BOTH,
    expand=True
)

bottom_frame = tk.Frame(root)
bottom_frame.pack(
    fill=tk.X,
    padx=14,
    pady=(0, 14)
)

question_entry = tk.Entry(
    bottom_frame,
    font=("Arial", 11)
)
question_entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=7,
    padx=(0, 8)
)

ask_button = tk.Button(
    bottom_frame,
    text="Ask AI",
    command=ask_question,
    width=12
)
ask_button.pack(side=tk.RIGHT)

question_entry.bind(
    "<Return>",
    lambda event: ask_question()
)

root.mainloop()
