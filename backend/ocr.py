import argparse
import json
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image, ImageOps

DEFAULT_LANG = "heb+eng"
DEFAULT_ROTATIONS = (0, 90, 180, 270)


def extract_text_regions(
    image_path: str | Path,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 6,
) -> list[dict]:
    """Run OCR over multiple rotations and return positioned text regions."""
    image = Image.open(image_path)
    return extract_text_regions_from_image(
        image,
        lang=lang,
        rotations=rotations,
        min_confidence=min_confidence,
        min_text_length=min_text_length,
        psm=psm,
    )


def extract_text_regions_from_image(
    image: Image.Image,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 6,
) -> list[dict]:
    """Run OCR over multiple rotations on a loaded image."""
    results = []

    for rotation in rotations:
        rotated = _prepare_image(image.rotate(rotation, expand=True))
        dataframe = pytesseract.image_to_data(
            rotated,
            lang=lang,
            output_type=pytesseract.Output.DATAFRAME,
            config=f"--psm {psm}",
        )
        results.extend(
            _rows_from_dataframe(
                dataframe,
                rotation,
                min_confidence,
                min_text_length,
            )
        )

    return results


def extract_tiled_text_regions(
    image_path: str | Path,
    rows: int,
    cols: int,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 6,
) -> list[dict]:
    """Split the image into tiles, OCR each tile, and keep tile metadata."""
    image = Image.open(image_path)
    width, height = image.size
    all_regions = []

    for row in range(rows):
        for col in range(cols):
            left = round(col * width / cols)
            top = round(row * height / rows)
            right = round((col + 1) * width / cols)
            bottom = round((row + 1) * height / rows)
            tile = image.crop((left, top, right, bottom))
            tile_regions = extract_text_regions_from_image(
                tile,
                lang=lang,
                rotations=rotations,
                min_confidence=min_confidence,
                min_text_length=min_text_length,
                psm=psm,
            )

            for region in tile_regions:
                region["tile_row"] = row
                region["tile_col"] = col
                region["tile"] = f"{row},{col}"
                if region["rotation"] == 0:
                    region["left"] += left * 2
                    region["top"] += top * 2
                all_regions.append(region)

    return all_regions


def extract_text_lines(
    image_path: str | Path,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 6,
) -> list[dict]:
    """Run OCR and group positioned word rows into compact text lines."""
    regions = extract_text_regions(
        image_path,
        lang=lang,
        rotations=rotations,
        min_confidence=min_confidence,
        min_text_length=min_text_length,
        psm=psm,
    )
    groups = {}

    for region in regions:
        key = (region["rotation"], region["block_num"], region["line_num"])
        groups.setdefault(key, []).append(region)

    lines = []
    for (rotation, block_num, line_num), words in groups.items():
        sorted_words = sorted(words, key=lambda word: word["left"])
        text = " ".join(word["text"] for word in sorted_words)
        left = min(word["left"] for word in sorted_words)
        top = min(word["top"] for word in sorted_words)
        right = max(word["left"] + word["width"] for word in sorted_words)
        bottom = max(word["top"] + word["height"] for word in sorted_words)
        confidence = sum(word["confidence"] for word in sorted_words) / len(sorted_words)

        lines.append(
            {
                "text": text,
                "confidence": round(confidence, 2),
                "rotation": rotation,
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "block_num": block_num,
                "line_num": line_num,
                "word_count": len(sorted_words),
            }
        )

    return sorted(lines, key=lambda line: (line["rotation"], line["top"], line["left"]))


def extract_text_clusters(
    image_path: str | Path,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    top_gap: int = 180,
    psm: int = 6,
) -> list[dict]:
    """Group nearby OCR lines into candidate book/spine text clusters."""
    lines = extract_text_lines(
        image_path,
        lang=lang,
        rotations=rotations,
        min_confidence=min_confidence,
        min_text_length=min_text_length,
        psm=psm,
    )
    clusters = []

    for rotation in rotations:
        rotation_lines = [line for line in lines if line["rotation"] == rotation]
        rotation_lines = sorted(rotation_lines, key=lambda line: (line["top"], line["left"]))

        current = []
        for line in rotation_lines:
            if not current:
                current = [line]
                continue

            previous = current[-1]
            same_band = abs(line["top"] - previous["top"]) <= top_gap
            if same_band:
                current.append(line)
            else:
                clusters.append(_cluster_from_lines(current))
                current = [line]

        if current:
            clusters.append(_cluster_from_lines(current))

    return sorted(clusters, key=lambda cluster: (cluster["rotation"], cluster["top"], cluster["left"]))


def extract_dbscan_clusters(
    image_path: str | Path,
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 11,
    x_distance: int = 450,
    y_distance: int = 80,
    tiles: tuple[int, int] | None = None,
) -> list[dict]:
    """Cluster OCR words by position using DBSCAN."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to use DBSCAN clustering.") from exc

    if tiles:
        regions = extract_tiled_text_regions(
            image_path,
            rows=tiles[0],
            cols=tiles[1],
            lang=lang,
            rotations=rotations,
            min_confidence=min_confidence,
            min_text_length=min_text_length,
            psm=psm,
        )
    else:
        regions = extract_text_regions(
            image_path,
            lang=lang,
            rotations=rotations,
            min_confidence=min_confidence,
            min_text_length=min_text_length,
            psm=psm,
        )
    clusters = []

    for rotation in rotations:
        words = [region for region in regions if region["rotation"] == rotation]
        if not words:
            continue

        features = [
            [
                word.get("tile_row", 0) * 1000,
                word.get("tile_col", 0) * 1000,
                (word["left"] + word["width"] / 2) / x_distance,
                (word["top"] + word["height"] / 2) / y_distance,
            ]
            for word in words
        ]
        labels = DBSCAN(eps=1.0, min_samples=1).fit_predict(features)

        for label in sorted(set(labels)):
            cluster_words = [word for word, word_label in zip(words, labels) if word_label == label]
            clusters.append(_cluster_from_words(cluster_words))

    return sorted(clusters, key=lambda cluster: (cluster["rotation"], cluster["top"], cluster["left"]))


def extract_combined_dbscan_clusters(
    image_path: str | Path,
    tiles_also: tuple[int, int],
    lang: str = DEFAULT_LANG,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    min_confidence: float = 20,
    min_text_length: int = 2,
    psm: int = 11,
    x_distance: int = 450,
    y_distance: int = 80,
) -> list[dict]:
    """Run DBSCAN OCR on the full image and on tiles, then dedupe fragments."""
    full_clusters = extract_dbscan_clusters(
        image_path,
        lang=lang,
        rotations=rotations,
        min_confidence=min_confidence,
        min_text_length=min_text_length,
        psm=psm,
        x_distance=x_distance,
        y_distance=y_distance,
    )
    tiled_clusters = extract_dbscan_clusters(
        image_path,
        lang=lang,
        rotations=rotations,
        min_confidence=min_confidence,
        min_text_length=min_text_length,
        psm=psm,
        x_distance=x_distance,
        y_distance=y_distance,
        tiles=tiles_also,
    )

    for cluster in full_clusters:
        cluster["source"] = "full"
    for cluster in tiled_clusters:
        cluster["source"] = "tile"

    return _dedupe_clusters(full_clusters + tiled_clusters)


def _cluster_from_lines(lines: list[dict]) -> dict:
    sorted_lines = sorted(lines, key=lambda line: line["left"])
    left = min(line["left"] for line in sorted_lines)
    top = min(line["top"] for line in sorted_lines)
    right = max(line["left"] + line["width"] for line in sorted_lines)
    bottom = max(line["top"] + line["height"] for line in sorted_lines)
    word_count = sum(line["word_count"] for line in sorted_lines)
    confidence = sum(line["confidence"] * line["word_count"] for line in sorted_lines) / word_count

    return {
        "text": " | ".join(line["text"] for line in sorted_lines),
        "confidence": round(confidence, 2),
        "rotation": sorted_lines[0]["rotation"],
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "line_count": len(sorted_lines),
        "word_count": word_count,
    }


def _cluster_from_words(words: list[dict]) -> dict:
    rotation = words[0]["rotation"]
    sorted_words = sorted(
        words,
        key=lambda word: (word["block_num"], word["line_num"], word["word_num"]),
    )
    left = min(word["left"] for word in sorted_words)
    top = min(word["top"] for word in sorted_words)
    right = max(word["left"] + word["width"] for word in sorted_words)
    bottom = max(word["top"] + word["height"] for word in sorted_words)
    confidence = sum(word["confidence"] for word in sorted_words) / len(sorted_words)

    return {
        "text": " ".join(word["text"] for word in sorted_words),
        "confidence": round(confidence, 2),
        "rotation": rotation,
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "word_count": len(sorted_words),
        "tile": sorted_words[0].get("tile"),
    }


def _dedupe_clusters(clusters: list[dict]) -> list[dict]:
    deduped = {}
    for cluster in clusters:
        key = _normalize_cluster_text(cluster["text"])
        existing = deduped.get(key)
        if existing is None or cluster["confidence"] > existing["confidence"]:
            deduped[key] = cluster

    return sorted(
        deduped.values(),
        key=lambda cluster: (
            cluster.get("source", ""),
            cluster["rotation"],
            cluster.get("tile") or "",
            cluster["top"],
            cluster["left"],
        ),
    )


def _normalize_cluster_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _prepare_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale)
    width, height = contrasted.size
    return contrasted.resize((width * 2, height * 2))


def _rows_from_dataframe(
    dataframe: pd.DataFrame,
    rotation: int,
    min_confidence: float,
    min_text_length: int,
) -> list[dict]:
    rows = []
    for row in dataframe.itertuples(index=False):
        text = getattr(row, "text", None)
        confidence = getattr(row, "conf", -1)

        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if len(text) < min_text_length:
            continue
        if confidence < min_confidence:
            continue

        rows.append(
            {
                "text": text,
                "confidence": float(confidence),
                "rotation": rotation,
                "left": int(getattr(row, "left")),
                "top": int(getattr(row, "top")),
                "width": int(getattr(row, "width")),
                "height": int(getattr(row, "height")),
                "block_num": int(getattr(row, "block_num")),
                "line_num": int(getattr(row, "line_num")),
                "word_num": int(getattr(row, "word_num")),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR research on a book photo.")
    parser.add_argument("image_path")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument(
        "--rotations",
        default="0,90,180,270",
        help="Comma-separated rotations to test, for example 270 or 0,270.",
    )
    parser.add_argument("--min-confidence", type=float, default=20)
    parser.add_argument("--min-text-length", type=int, default=2)
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument(
        "--format",
        choices=("words", "lines", "clusters", "dbscan", "text"),
        default="lines",
        help="Print word rows, grouped line rows, clusters, or compact text lines.",
    )
    parser.add_argument("--top-gap", type=int, default=180)
    parser.add_argument("--x-distance", type=int, default=450)
    parser.add_argument("--y-distance", type=int, default=80)
    parser.add_argument(
        "--tiles",
        help="Optional rows,cols tiling before OCR, for example 3,1 or 2,3.",
    )
    parser.add_argument(
        "--tiles-also",
        help="For DBSCAN, run full image plus these rows,cols tiles and dedupe.",
    )
    parser.add_argument(
        "--clean-openai",
        action="store_true",
        help="Send OCR output to OpenAI for text-only book cleanup.",
    )
    args = parser.parse_args()
    rotations = tuple(int(rotation.strip()) for rotation in args.rotations.split(","))
    tiles = None
    if args.tiles:
        tile_parts = tuple(int(part.strip()) for part in args.tiles.split(","))
        if len(tile_parts) != 2:
            raise ValueError("--tiles must look like rows,cols")
        tiles = tile_parts
    tiles_also = None
    if args.tiles_also:
        tile_parts = tuple(int(part.strip()) for part in args.tiles_also.split(","))
        if len(tile_parts) != 2:
            raise ValueError("--tiles-also must look like rows,cols")
        tiles_also = tile_parts

    if args.format == "words":
        output = extract_text_regions(
            args.image_path,
            lang=args.lang,
            rotations=rotations,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            psm=args.psm,
        )
    elif args.format in {"lines", "text"}:
        output = extract_text_lines(
            args.image_path,
            lang=args.lang,
            rotations=rotations,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            psm=args.psm,
        )
    elif args.format == "clusters":
        output = extract_text_clusters(
            args.image_path,
            lang=args.lang,
            rotations=rotations,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            top_gap=args.top_gap,
            psm=args.psm,
        )
    elif tiles_also:
        output = extract_combined_dbscan_clusters(
            args.image_path,
            tiles_also=tiles_also,
            lang=args.lang,
            rotations=rotations,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            psm=args.psm,
            x_distance=args.x_distance,
            y_distance=args.y_distance,
        )
    else:
        output = extract_dbscan_clusters(
            args.image_path,
            lang=args.lang,
            rotations=rotations,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length,
            psm=args.psm,
            x_distance=args.x_distance,
            y_distance=args.y_distance,
            tiles=tiles,
        )

    if args.clean_openai:
        try:
            from backend.ai import clean_ocr_book_fragments
        except ModuleNotFoundError:
            from ai import clean_ocr_book_fragments

        cleaned = clean_ocr_book_fragments(_cleanup_fragments(output))
        print(json.dumps(cleaned, ensure_ascii=False, indent=2))
    elif args.format == "text":
        for line in output:
            print(
                f'{line["rotation"]:>3} '
                f'{line["confidence"]:>5.1f} '
                f'{line["word_count"]:>2} '
                f'{line["text"]}'
            )
    elif args.format == "clusters":
        for cluster in output:
            print(
                f'{cluster["rotation"]:>3} '
                f'{cluster["confidence"]:>5.1f} '
                f'{cluster["line_count"]:>2} '
                f'{cluster["word_count"]:>2} '
                f'{cluster["text"]}'
            )
    elif args.format == "dbscan":
        for cluster in output:
            print(
                f'{cluster["rotation"]:>3} '
                f'{cluster["confidence"]:>5.1f} '
                f'{cluster["word_count"]:>2} '
                f'{cluster["text"]}'
            )
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _cleanup_fragments(clusters: list[dict]) -> list[dict]:
    return [
        cluster
        for cluster in clusters
        if cluster["word_count"] >= 2 or cluster["confidence"] >= 80
    ]


if __name__ == "__main__":
    main()
