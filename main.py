from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
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

@app.get("/")
def home():
    return {"message": "Absolute Vector Text Stripper is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open the PDF natively as a pure vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        p_width = page.rect.x1
        p_height = page.rect.y1
        
        # Strictly define the text layer boundary of the watermark corner
        # This targets only the absolute bottom-right spot where the watermark sits
        watermark_rect = fitz.Rect(p_width - 150, p_height - 42, p_width - 15, p_height - 12)
        
        # fill=None guarantees that no solid color overlay block is created
        page.add_redact_annot(watermark_rect, fill=None)
        
        # THE CRITICAL CORE FIX:
        # images=0 (fitz.PDF_REDACT_IMAGE_NONE) -> Protects all background images
        # graphics=0 (fitz.PDF_REDACT_GRAPHICS_NONE) -> TOTAL PROTECTION for blueprint grids and vector lines!
        # This forces the engine to ONLY strip the text glyphs, leaving the grid lines 100% untouched.
        page.apply_redactions(images=0, graphics=0)
        
    output_stream = io.BytesIO()
    doc.save(output_stream, garbage=4, deflate=True)
    doc.close()
    output_stream.seek(0)
    
    gc.collect()
    
    return StreamingResponse(
        output_stream, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"}
    )
    
