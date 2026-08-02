from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are a technical support assistant.

Answer only from the provided documentation.

Give a concise, direct answer.
Do not repeat the title, description, or source metadata.
Preserve numbered steps when the documentation contains a procedure.
Do not add unsupported information.

If the documentation does not contain the answer, say:
"I couldn't find enough information in the provided documentation."
""".strip()


def generate_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or whichever model you use
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Documentation:

{context}

Question:

{question}
""",
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("LLM returned no content.")

    return content
