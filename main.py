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
    return {"message": "Vector-Level Professional PDF Cleaner is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    # Read incoming PDF bytes directly into memory
    pdf_bytes = await file.read()
    
    # Open the PDF natively as a vector document (No image conversion!)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        # Search for the exact string coordinates on the native text layer
        text_instances = page.search_for("NotebookLM")
        
        for rect in text_instances:
            # CRITICAL PRO FIX: fill=None makes the redaction boundary completely transparent.
            # It surgically strips out the text characters directly from the PDF stream,
            # leaving underlying background drawings, grids, and themes 100% untouched.
            page.add_redact_annot(rect, fill=None)
        
        # Apply the text layer redactions natively onto the PDF vector stream
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        
    # Compress and polish the output document to maintain high professional standards
    output_stream = io.BytesIO()
    doc.save(output_stream, garbage=4, deflate=True)
    doc.close()
    output_stream.seek(0)
    
    # Instant memory cleanup
    gc.collect()
    
    return StreamingResponse(
        output_stream, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"}
    )
    
