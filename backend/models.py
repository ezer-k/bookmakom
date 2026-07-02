from pydantic import BaseModel


class BookCreate(BaseModel):
    title: str | None = None
    author: str | None = None
    lat: float
    lng: float
    image_url: str | None = None
    google_books_link: str | None = None


class BookMarker(BookCreate):
    id: int
    spotted_at: str | None = None


class Meta(BaseModel):
    total_uploads: int


class UploadResponse(BaseModel):
    status: str
    books_added: list[BookMarker]
