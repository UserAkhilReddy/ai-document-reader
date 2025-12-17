from transformers import pipeline

# Load model once (important)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def answer_question(question: str, context_chunks: list[str]) -> str:
    """
    Takes a question and relevant text chunks
    Returns a concise answer
    """

    context = " ".join(context_chunks)

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    result = qa_pipeline(prompt, max_length=256, do_sample=False)

    return result[0]["generated_text"]
