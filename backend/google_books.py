import os

import requests

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"


def search_book(title: str | None, author: str | None = None) -> dict | None:
    """Return the best Google Books match for a title/author pair."""
    query = " ".join(part for part in [title, author] if part)
    if not query:
        return None

    try:
        params = {"q": query, "maxResults": 1}
        if api_key := os.getenv("GOOGLE_BOOKS_API_KEY"):
            params["key"] = api_key

        response = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    items = response.json().get("items", [])
    if not items:
        return None

    volume_info = items[0].get("volumeInfo", {})
    authors = volume_info.get("authors") or []

    return {
        "title": volume_info.get("title"),
        "author": authors[0] if authors else None,
        "google_books_link": volume_info.get("infoLink"),
    }
