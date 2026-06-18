from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF >= 1.23
import io
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NotebookLM Watermark Remover", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Watermark geometry constants ──────────────────────────────────────────────
# NotebookLM places its watermark in the bottom-right corner of every page.
# These values are in *image pixels* relative to the embedded raster image.
WM_H: int = 55   # watermark zone height (pixels from bottom edge)
WM_W: int = 215  # watermark zone width  (pixels from right edge)

# Pixels darker than this threshold are considered part of the watermark text.
# Background grid is typically ~220-250; watermark text is ~80-210.
WM_THRESHOLD: int = 215


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_png_lossless(img_bgr: np.ndarray) -> bytes:
    """
    Encode a BGR NumPy image to PNG bytes with zero pixel loss.

    Uses Pillow (not OpenCV imencode) because Pillow guarantees a
    perfectly lossless PNG round-trip.  OpenCV and fitz.Pixmap both
    introduce small colour shifts during internal colour-space handling.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def build_text_mask(zone_gray: np.ndarray) -> np.ndarray:
    """
    Return a binary mask that covers only the watermark text pixels.

    Steps
    -----
    1. Simple binary threshold  – isolates dark text from light background.
    2. Morphological dilation   – expands by 1-2 px to erase anti-aliased
       fringe pixels that threshold alone leaves behind.
    """
    _, mask = cv2.threshold(zone_gray, WM_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def sample_background(img: np.ndarray, wm_y1: int, wm_x1: int) -> np.ndarray:
    """
    Sample the background texture from the region directly ABOVE the watermark.

    This guarantees the fill uses the *exact* grid / gradient pattern that
    should be visible in the watermark zone, making the removal invisible.

    If the page has content too close to the top of the watermark area, we
    tile whatever reference we can find so the fill always has the right shape.
    """
    ref_y1 = max(0, wm_y1 - WM_H)
    ref = img[ref_y1:wm_y1, wm_x1 : wm_x1 + WM_W].copy()

    if ref.shape[0] == 0:          # degenerate edge case
        ref = np.full((WM_H, WM_W, 3), 245, dtype=np.uint8)
    elif ref.shape[0] < WM_H:      # not enough rows → tile vertically
        reps = int(np.ceil(WM_H / ref.shape[0]))
        ref = np.tile(ref, (reps, 1, 1))

    return ref[:WM_H, :WM_W]       # exact crop to watermark zone size


# ── Core removal logic ────────────────────────────────────────────────────────

def remove_notebooklm_watermark_from_page(
    page: fitz.Page,
    doc: fitz.Document,
) -> bool:
    """
    Remove the NotebookLM watermark from a single PDF page.

    The NotebookLM watermark exists as TWO independent layers:

    Layer A  –  Raster layer
        The "🎧 NotebookLM" text is baked directly into the embedded
        page image (PNG/JPEG XObject stored in the PDF cross-reference
        table).  We must edit the image pixels in-place.

        Fix:
          • Extract the image from its PDF xref.
          • Compute a pixel mask for the watermark text area.
          • Fill masked pixels with a reference texture sampled from
            directly above the watermark zone (same column range, same
            background pattern).
          • Re-inject the edited image with page.replace_image(), which
            is the only PyMuPDF API that correctly handles image xrefs
            (update_stream() bypasses the image filter pipeline and
            causes colour corruption).

    Layer B  –  Vector / drawing layer
        A series of filled coloured rectangles arranged in a triangular
        fade pattern.  These are written as PDF content-stream drawing
        operators (moveto / lineto / fill) by NotebookLM into the LAST
        content stream of each page.

        Fix:
          • Overwrite that content stream with an empty byte string.

    Returns True if any modification was made, False otherwise.
    """
    modified = False

    # ── Layer B: erase vector drawing stream ──────────────────────────────
    contents = page.get_contents()
    if len(contents) >= 4:
        # NotebookLM always appends its vector ops as the last stream
        last_xref = contents[-1]
        stream_data = doc.xref_stream(last_xref)

        # Only blank it when it actually contains watermark drawing ops.
        # A genuine watermark stream is large (> 200 bytes) and uses the
        # characteristic fill colour  0.956 0.941 0.902  (the beige tint).
        if len(stream_data) > 200 and b"0.9568627450980393" in stream_data:
            doc.update_stream(last_xref, b"")
            modified = True
            logger.debug("Page %d: vector layer cleared", page.number + 1)

    # ── Layer A: patch raster image ────────────────────────────────────────
    page_images = page.get_images(full=True)
    if not page_images:
        return modified

    # The main background is always the largest image on the page
    img_xref = max(page_images, key=lambda x: x[2] * x[3])[0]

    try:
        base_image = doc.extract_image(img_xref)
    except Exception as exc:
        logger.warning("Page %d: could not extract image xref %d – %s",
                       page.number + 1, img_xref, exc)
        return modified

    raw = np.frombuffer(base_image["image"], np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Page %d: cv2.imdecode returned None", page.number + 1)
        return modified

    img_h, img_w = img.shape[:2]
    wm_y1 = img_h - WM_H
    wm_x1 = img_w - WM_W

    if wm_y1 < 0 or wm_x1 < 0:
        logger.warning("Page %d: image too small for watermark zone", page.number + 1)
        return modified

    zone = img[wm_y1:img_h, wm_x1:img_w].copy()
    zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    text_mask = build_text_mask(zone_gray)
    text_pixel_count = int(cv2.countNonZero(text_mask))

    if text_pixel_count < 5:
        logger.debug("Page %d: no watermark pixels found, skipping raster pass",
                     page.number + 1)
        return modified

    # Fill text pixels with the background texture from above
    bg_fill = sample_background(img, wm_y1, wm_x1)
    mask_3ch = cv2.merge([text_mask, text_mask, text_mask])
    cleaned_zone = np.where(mask_3ch > 0, bg_fill, zone).astype(np.uint8)
    img[wm_y1:img_h, wm_x1:img_w] = cleaned_zone

    # Re-inject the edited image using replace_image (correct PyMuPDF API)
    png_bytes = encode_png_lossless(img)
    page.replace_image(img_xref, stream=png_bytes)
    modified = True

    logger.info(
        "Page %d: watermark removed  (text px=%d, image=%dx%d)",
        page.number + 1, text_pixel_count, img_w, img_h,
    )

    # Free large arrays immediately; we process pages sequentially
    del img, zone, zone_gray, text_mask, bg_fill, mask_3ch, cleaned_zone, raw
    gc.collect()

    return modified


# ── FastAPI routes ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "NotebookLM Watermark Remover",
        "version": "2.0.0",
    }


@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    """
    Accept a PDF (upload), strip all NotebookLM watermarks, return clean PDF.

    The endpoint processes every page in the document.  Pages that do not
    carry the watermark are left completely untouched.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")

    pages_modified = 0
    for page in doc:
        if remove_notebooklm_watermark_from_page(page, doc):
            pages_modified += 1

    logger.info("Processing complete: %d/%d pages modified", pages_modified, len(doc))

    output = io.BytesIO()
    doc.save(
        output,
        garbage=4,      # remove unreferenced objects
        deflate=True,   # recompress streams
        clean=True,     # sanitise content streams
    )
    doc.close()
    output.seek(0)
    gc.collect()

    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=cleaned_document.pdf",
            "X-Pages-Modified": str(pages_modified),
        },
    )
