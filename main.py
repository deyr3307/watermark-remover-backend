from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import fitz  # PyMuPDF
import io

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
    return {"message": "Masterclass PDF Watermark Remover is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open PDF from memory
    input_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    output_pdf = fitz.open()
    
    for page_num in range(len(input_pdf)):
        page = input_pdf[page_num]
        
        # 1. SMART SEARCH: Find exact coordinates of "NotebookLM" text on this specific page
        text_instances = page.search_for("NotebookLM")
        
        # Render page to high-quality image matrix
        pix = page.get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate coordinate scaling factor between PDF points and Image pixels
        zoom_x = w / page.rect.width
        zoom_y = h / page.rect.height
        
        # 2. PRECISE MASKING: If the text is found, mask ONLY the exact letter bounds
        if text_instances:
            for rect in text_instances:
                x0 = max(0, int(rect.x0 * zoom_x) - 4)
                y0 = max(0, int(rect.y0 * zoom_y) - 4)
                x1 = min(w, int(rect.x1 * zoom_x) + 4)
                y1 = min(h, int(rect.y1 * zoom_y) + 4)
                
                # Fill only the text rectangle with pure white on mask
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        else:
            # Fallback: If text layer is flattened, target the exact bottom-right corner safely
            cv2.rectangle(mask, (int(w - 180), int(h - 45)), (w, h), 255, -1)
            
        # 3. INPAINTING: Reconstruct the background seamlessly regardless of color/theme
        cleaned_img = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        # Encode back to high-quality PDF page structure
        _, img_encoded = cv2.imencode('.jpg', cleaned_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_pdf_bytes = fitz.image_to_pdf(img_encoded.tobytes())
        img_pdf = fitz.open("pdf", img_pdf_bytes)
        
        output_pdf.insert_pdf(img_pdf)
        
    # Save the polished document to stream
    output_stream = io.BytesIO()
    output_pdf.save(output_stream)
    output_stream.seek(0)
    
    return StreamingResponse(
        output_stream, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"}
        )
        
