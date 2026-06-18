from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
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
    return {"message": "High-Power Pixel-Level Watermark Remover is Ready!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    input_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    output_pdf = fitz.open()
    
    for page_num in range(len(input_pdf)):
        page = input_pdf[page_num]
        text_instances = page.search_for("NotebookLM")
        
        # Render page to image bytes safely using PNG
        pix = page.get_pixmap(dpi=120)
        img_bytes = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        zoom_x = w / page.rect.width
        zoom_y = h / page.rect.height
        
        if text_instances:
            for rect in text_instances:
                # Get coordinates with a tiny 2px safety padding
                x0 = max(0, int(rect.x0 * zoom_x) - 2)
                y0 = max(0, int(rect.y0 * zoom_y) - 2)
                x1 = min(w, int(rect.x1 * zoom_x) + 2)
                y1 = min(h, int(rect.y1 * zoom_y) + 2)
                
                # High-Power Step: Extract ONLY the dark text strokes, preserve the grid lines
                crop = img[y0:y1, x0:x1]
                if crop.size > 0:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Isolate dark text color vectors from the light blueprint grid
                    _, thresh_crop = cv2.threshold(gray_crop, 135, 255, cv2.THRESH_BINARY_INV)
                    
                    # Smoothly dilate by 1px to ensure no text ghosting remains
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thresh_crop = cv2.dilate(thresh_crop, kernel, iterations=1)
                    
                    # Merge into the global mask array
                    mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], thresh_crop)
        else:
            # High-Power Fallback: Target the corner zone if the PDF layer is flat
            x0, y0, x1, y1 = int(w - 160), int(h - 50), w, h
            crop = img[y0:y1, x0:x1]
            if crop.size > 0:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh_crop = cv2.threshold(gray_crop, 135, 255, cv2.THRESH_BINARY_INV)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                thresh_crop = cv2.dilate(thresh_crop, kernel, iterations=1)
                mask[y0:y1, x0:x1] = thresh_crop
                
        # Inpaint exactly on the micro-level letter tracks with a tight radius (3)
        # This keeps the background grid vectors running perfectly through the letters
        cleaned_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Encode back to PDF layer structure with premium compression
        _, img_encoded = cv2.imencode('.jpg', cleaned_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_pdf_bytes = fitz.image_to_pdf(img_encoded.tobytes())
        
        img_pdf = fitz.open("pdf", img_pdf_bytes)
        output_pdf.insert_pdf(img_pdf)
        
        img_pdf.close()
        del img, mask, cleaned_img
        
    input_pdf.close()
    
    output_stream = io.BytesIO()
    output_pdf.save(output_stream)
    output_pdf.close()
    output_stream.seek(0)
    
    gc.collect()
    
    return StreamingResponse(
        output_stream, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"}
                    )
    
