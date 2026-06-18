from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import io
import re
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB cap, প্রয়োজনে বদলাতে পারো

@app.get("/")
def home():
    return {"message": "Absolute Pure-Vector Stream Stripper is Running!"}


@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open file as a valid PDF.")

    if doc.is_encrypted:
        doc.close()
        raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported.")

    try:
        for page in doc:
            page.clean_contents()
            p_width = page.rect.x1

            for xref in page.get_contents():
                stream_data = doc.xref_stream(xref)
                if stream_data is None:
                    continue

                blocks = re.findall(rb"BT[\s\S]*?ET", stream_data)

                for block in blocks:
                    if block not in stream_data:
                        # আগেই অন্য কোনো identical block-এর সাথে মুছে গেছে
                        continue

                    is_watermark = False

                    if b"notebook" in block.lower():
                        is_watermark = True

                    # Tm operator: a b c d e f Tm -> আমাদের লাগবে e, f (translation/position)
                    tm_match = re.search(
                        rb"([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+([0-9.+-]+)\s+Tm",
                        block,
                    )
                    if tm_match:
                        try:
                            x = float(tm_match.group(5))
                            y = float(tm_match.group(6))
                            if x > (p_width - 190) and y < 65:
                                is_watermark = True
                        except ValueError:
                            pass

                    # Td operator: tx ty Td -> এটা সত্যিই ২টা মাত্র operand নেয়
                    td_match = re.search(
                        rb"([0-9.+-]+)\s+([0-9.+-]+)\s+Td",
                        block,
                    )
                    if td_match:
                        try:
                            x = float(td_match.group(1))
                            y = float(td_match.group(2))
                            if (x > (p_width - 190) or x > 400) and y < 65:
                                is_watermark = True
                        except ValueError:
                            pass

                    if is_watermark:
                        stream_data = stream_data.replace(block, b"")

                doc.update_stream(xref, stream_data)

        output_stream = io.BytesIO()
        doc.save(output_stream, garbage=4, deflate=True)
    except Exception as e:
        doc.close()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        doc.close()

    output_stream.seek(0)
    gc.collect()

    return StreamingResponse(
        output_stream,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"},
                        )
