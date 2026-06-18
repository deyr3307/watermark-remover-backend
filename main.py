from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
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

@app.get("/")
def home():
    return {"message": "Professional Dual-Engine Vector Cleaner is Live!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open PDF as a native vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        # ENGINE 1: Lossless Vector Content Stream Cleaning
        page.clean_contents()  # Decompress and normalize all page streams
        
        if page.get_contents():
            xref = page.get_contents()[0]
            stream_data = doc.xref_stream(xref)
            
            # Regex to find and completely remove the exact BT...ET text block containing NotebookLM
            watermark_pattern = b"BT[\s\S]*?NotebookLM[\s\S]*?ET"
            modified_stream = re.sub(watermark_pattern, b"", stream_data)
            
            # If Engine 1 successfully stripped the text object, save and move to next page
            if len(modified_stream) < len(stream_data):
                doc.update_stream(xref, modified_stream)
                continue  # 100% Lossless vector removal done for this page!
        
        # ENGINE 2: Ultra-High DPI Surgical Patch (Fallback for flattened layers)
        p_width = page.rect.x1
        p_height = page.rect.y1
        
        # Strictly define the micro-boundary box of the watermark corner
        clip_rect = fitz.Rect(p_width - 150, p_height - 42, p_width, p_height)
        
        # Render ONLY the tiny corner at an ultra-sharp 450 DPI to prevent any pixelation or blur
        pix = page.get_pixmap(clip=clip_rect, dpi=450)
        img_bytes = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None and img.size > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Isolate the text stroke paths from the blueprint grid background
            _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Inpaint exactly over the text tracks using tight surrounding grid contexts
            cleaned_crop = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
            _, img_encoded = cv2.imencode('.png', cleaned_crop)
            
            # Stamp the perfectly clear high-res patch natively back onto the PDF stream layout
            page.insert_image(clip_rect, stream=img_encoded.tobytes())
            
            del img, mask, cleaned_crop
            
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
    
