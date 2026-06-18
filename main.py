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
    return {"message": "Grid-Reconstruction Vector Cleaner is Natively Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open the PDF natively as a pure vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        p_width = page.rect.x1
        p_height = page.rect.y1
        
        # Define the exact bounding box around the bottom-right watermark zone
        target_rect = fitz.Rect(p_width - 165, p_height - 42, p_width - 10, p_height - 5)
        
        # 1. DYNAMIC BACKGROUND COLOR SAMPLING
        # Sample a 1x1 pixel slightly outside the zone to capture the exact underlying page theme color
        sample_rect = fitz.Rect(p_width - 170, p_height - 50, p_width - 169, p_height - 49)
        pix = page.get_pixmap(clip=sample_rect, dpi=10)
        
        if pix and len(pix.samples) >= 3:
            bg_color = (pix.samples[0] / 255.0, pix.samples[1] / 255.0, pix.samples[2] / 255.0)
        else:
            bg_color = (1.0, 1.0, 1.0)  # Fallback to absolute white if sampling fails
            
        # 2. ADVANCED VECTOR GRID LINE DETECTION
        vertical_xs = set()
        horizontal_ys = set()
        grid_color = (0.85, 0.85, 0.85)  # Default light gray fallback
        grid_width = 0.5
        
        drawings = page.get_drawings()
        for draw in drawings:
            stroke_color = draw.get("color", None)
            w = draw.get("width", 0.5)
            
            for item in draw.get("items", []):
                if item[0] == "l":  # Identify pure native straight line vectors
                    p1, p2 = item[1], item[2]
                    
                    # Track vertical grid lines that intersect the watermark box coordinates
                    if abs(p1.x - p2.x) < 0.1:
                        if min(p1.y, p2.y) < target_rect.y0 and max(p1.y, p2.y) > target_rect.y1:
                            vertical_xs.add(p1.x)
                            if stroke_color:
                                grid_color = stroke_color
                                grid_width = w
                                
                    # Track horizontal grid lines that intersect the watermark box coordinates
                    elif abs(p1.y - p2.y) < 0.1:
                        if min(p1.x, p2.x) < target_rect.x0 and max(p1.x, p2.x) > target_rect.x1:
                            horizontal_ys.add(p1.y)
                            if stroke_color:
                                grid_color = stroke_color
                                grid_width = w
                                
        # 3. SEAMLESS RECONSTRUCTION OVERLAY
        # Conceal the text layer completely using a solid vector block matching the background theme
        page.draw_rect(target_rect, color=bg_color, fill=bg_color, overlay=True)
        
        # Redraw the missing vertical grid lines perfectly across the concealed block
        for x in vertical_xs:
            if target_rect.x0 <= x <= target_rect.x1:
                page.draw_line(fitz.Point(x, target_rect.y0), fitz.Point(x, target_rect.y1), 
                               color=grid_color, width=grid_width, overlay=True)
                
        # Redraw the missing horizontal grid lines perfectly across the concealed block
        for y in horizontal_ys:
            if target_rect.y0 <= y <= target_rect.y1:
                page.draw_line(fitz.Point(target_rect.x0, y), fitz.Point(target_rect.x1, y), 
                               color=grid_color, width=grid_width, overlay=True)
                
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
    
