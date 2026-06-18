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
    return {"message": "PDF Watermark Remover Backend is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    # Read the uploaded PDF file bytes
    pdf_bytes = await file.read()
    
    # Open the PDF document from memory
    input_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    output_pdf = fitz.open()
    
    # Process each page of the PDF one by one
    for page_num in range(len(input_pdf)):
        page = input_pdf[page_num]
        
        # Render the PDF page into a high-quality image matrix (DPI = 150)
        pix = page.get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Adjust color channels for OpenCV processing
        if pix.n == 4:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Target the exact bottom-right corner where the watermark sits
        crop_start_y = int(h - 55)
        crop_start_x = int(w - 200)
        
        watermark_zone = img[crop_start_y:h, crop_start_x:w]
        gray_zone = cv2.cvtColor(watermark_zone, cv2.COLOR_BGR2GRAY)
        
        # Extract the vector shape of the watermark letters
        _, text_mask = cv2.threshold(gray_zone, 110, 255, cv2.THRESH_BINARY_INV)
        mask[crop_start_y:h, crop_start_x:w] = text_mask
        
        # Clean the watermark using Inpainting
        cleaned_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Encode the cleaned image back to JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', cleaned_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Convert the processed image page back into a PDF structure
        img_pdf_bytes = fitz.image_to_pdf(img_encoded.tobytes())
        img_pdf = fitz.open("pdf", img_pdf_bytes)
        
        # Insert the cleaned page into our new output PDF document
        output_pdf.insert_pdf(img_pdf)
        
    # Save the complete combined clean PDF into memory
    output_stream = io.BytesIO()
    output_pdf.save(output_stream)
    output_stream.seek(0)
    
    # Stream the actual clear PDF file back to your Vercel frontend
    return StreamingResponse(
        output_stream, 
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cleaned_document.pdf"}
        )
    
