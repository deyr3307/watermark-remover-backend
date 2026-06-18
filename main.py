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
    return {"message": "Vector Shield PDF Cleaner is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open PDF as a pure native vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        # Locate the exact bounding box coordinates of the watermark text
        text_instances = page.search_for("NotebookLM")
        
        if text_instances:
            for rect in text_instances:
                # Draw a razor-sharp, pure white vector shield EXACTLY over the letters
                # color=(1,1,1) and fill=(1,1,1) represents pure solid white in PyMuPDF
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
        else:
            # High-Precision Fallback: If font encoding is masked, shield the exact bottom-right spot
            p_width = page.rect.x1
            p_height = page.rect.y1
            fallback_rect = fitz.Rect(p_width - 150, p_height - 42, p_width - 20, p_height - 15)
            page.draw_rect(fallback_rect, color=(1, 1, 1), fill=(1, 1, 1), overlay=True)
            
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
