import base64
import json
import os

from openai import OpenAI

VISION_PROMPT = """
You are extracting visible book text from a photo.
The image may contain book covers, spines, piles, shelves, boxes, or street stacks.
Scan the image systematically from top-left to bottom-right.
Return a book entry only when you can read text directly from a visible book.
Return at most 20 distinct books.
Do not repeat the same title.

For each book:
- Use only text that is visible in the image.
- Do not invent, translate, complete, or guess titles.
- Do not infer authors from memory.
- Partial titles are allowed, but keep only the readable visible words.
- If the author is not visibly printed, use null.
- Prefer missing a book over hallucinating a book that is not visibly readable.
- If you cannot read distinct visible book text, return {"books": []}.

Return only JSON that matches this shape:
{"books": [{"title": "The Trial", "author": "Franz Kafka"}]}
""".strip()

BOOKS_SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "author": {"type": ["string", "null"]},
                },
                "required": ["title", "author"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["books"],
    "additionalProperties": False,
}


def analyze_book_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    model: str | None = None,
) -> list[dict]:
    """Return detected books from an uploaded image."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    image_data = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_data}"

    client = OpenAI()
    response = client.responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
        max_output_tokens=2500,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "detected_books",
                "schema": BOOKS_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    return _parse_books(response.output_text)


def _parse_books(raw_text: str) -> list[dict]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        preview = raw_text[:500]
        raise ValueError(f"OpenAI returned invalid JSON: {exc}. Preview: {preview}") from exc

    books = parsed.get("books", parsed if isinstance(parsed, list) else [])

    clean_books = []
    seen_titles = set()
    for book in books:
        title = book.get("title")
        author = book.get("author")
        title_key = title.strip().lower() if isinstance(title, str) else None
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        if title or author:
            clean_books.append({"title": title, "author": author})

    return clean_books
