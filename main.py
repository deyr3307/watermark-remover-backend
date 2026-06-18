from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz
import numpy as np
from PIL import Image
import io
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# NotebookLM watermark position — measured from real exported PDFs.
# Stored as ratios of page dimensions so it works at any render DPI.
# These values were measured on NotebookLM slide-style exports (16:9, 1376x768 pt).
# If a different template is used, these may need updating.
WM_X_MIN_RATIO = 5205 / 5734
WM_X_MAX_RATIO = 5700 / 5734
WM_Y_MIN_RATIO = 3070 / 3200
WM_Y_MAX_RATIO = 3168 / 3200

RENDER_DPI = 200  # balance between quality and speed/memory


def remove_notebooklm_watermark(pdf_bytes: bytes) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if doc.is_encrypted:
        doc.close()
        raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported.")

    new_doc = fitz.open()

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        h, w, _ = img.shape

        x0 = int(WM_X_MIN_RATIO * w)
        x1 = int(WM_X_MAX_RATIO * w)
        y0 = int(WM_Y_MIN_RATIO * h)
        y1 = int(WM_Y_MAX_RATIO * h)

        # Clamp to image bounds just in case
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)

        if x1 > x0 and y1 > y0:
            # Sample background color from the right-side margin beyond the watermark.
            # This corner is always blank page margin in NotebookLM exports.
            right_margin = img[y0:y1, x1:w]

            if right_margin.size > 0 and right_margin.reshape(-1, 3).std() < 20:
                bg_color = right_margin.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            else:
                # Fallback: sample a strip below the watermark box
                below = img[y1:min(h, y1 + 15), x0:x1]
                if below.size > 0:
                    bg_color = below.reshape(-1, 3).mean(axis=0).astype(np.uint8)
                else:
                    # Last resort: use a typical NotebookLM background cream color
                    bg_color = np.array([245, 240, 228], dtype=np.uint8)

            img[y0:y1, x0:x1] = bg_color

        # Save cleaned image to memory buffer
        out_img = Image.fromarray(img)
        img_buf = io.BytesIO()
        out_img.save(img_buf, format="PNG")
        img_buf.seek(0)

        new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=img_buf.read())

    doc.close()

    out_buf = io.BytesIO()
    new_doc.save(out_buf, garbage=4, deflate=True)
    new_doc.close()
    out_buf.seek(0)
    gc.collect()

    return out_buf.read()


@app.get("/")
def home():
    return {"message": "NotebookLM Watermark Remover is running!"}


@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 50MB.")

    try:
        fitz.open(stream=pdf_bytes, filetype="pdf").close()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open as a valid PDF.")

    try:
        cleaned_bytes = remove_notebooklm_watermark(pdf_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(cleaned_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"},
    )
