from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import BookMarker, Meta, UploadResponse

app = FastAPI(title="BookMakom API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/books", response_model=list[BookMarker])
def list_books(q: str | None = None, sort: str = "date"):
    return []


@app.get("/meta", response_model=Meta)
def get_meta():
    return Meta(total_uploads=0)


@app.post("/upload", response_model=UploadResponse)
async def upload_books(
    lat: float = Form(...),
    lng: float = Form(...),
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    return UploadResponse(
        status="success",
        books_added=[
            BookMarker(
                id=1,
                spotted_at=None,
                lat=lat,
                lng=lng,
                title=None,
                author=None,
                image_url=None,
            )
        ],
    )
