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
    
    # STEP 1: Deep Core Stream Wiping (Removes text codes directly from binary layers)
    for x in range(1, doc.xref_length()):
        try:
            stream = doc.xref_stream(x)
            if stream and b"NotebookLM" in stream:
                stream = stream.replace(b"NotebookLM", b"          ")
                doc.update_stream(x, stream)
        except:
            pass

    for page in doc:
        page.clean_contents()
        p_width = page.rect.x1
        p_height = page.rect.y1
        
        # Locate the exact layout coordinates of the watermark text
        text_instances = page.search_for("NotebookLM")
        
        # If text is found, target those exact bounds. Otherwise use high-precision corner fallback
        targets = text_instances if text_instances else [fitz.Rect(p_width - 165, p_height - 42, p_width - 10, p_height - 5)]
        
        for rect in targets:
            # 2. DYNAMIC BACKGROUND SAMPLING
            # Sample a pixel just outside the rect to lock the page's exact color theme
            sample_rect = fitz.Rect(rect.x0 - 5, rect.y0 - 5, rect.x0 - 1, rect.y0 - 1)
            pix = page.get_pixmap(clip=sample_rect, dpi=10)
            
            if pix and len(pix.samples) >= 3:
                bg_color = (pix.samples[0] / 255.0, pix.samples[1] / 255.0, pix.samples[2] / 255.0)
            else:
                bg_color = (1.0, 1.0, 1.0)
                
            # Cover the text layer cleanly using a native solid vector block matching the background
            page.draw_rect(rect, color=bg_color, fill=bg_color, overlay=True)
            
            # 3. NATIVE VECTOR GRID LINE RECONSTRUCTION
            # Scan all straight lines on the page and redraw any segments chopped by the overlay block
            vertical_xs = set()
            horizontal_ys = set()
            grid_color = (0.85, 0.85, 0.85)  # Default fallback grid color
            grid_width = 0.5
            
            drawings = page.get_drawings()
            for draw in drawings:
                stroke_color = draw.get("color", None)
                w = draw.get("width", 0.5)
                
                for item in draw.get("items", []):
                    if item[0] == "l":  # Identify straight line vectors
                        p1, p2 = item[1], item[2]
                        
                        # Reconstruct vertical grid lines
                        if abs(p1.x - p2.x) < 0.1:
                            if min(p1.y, p2.y) <= rect.y0 and max(p1.y, p2.y) >= rect.y1:
                                if rect.x0 <= p1.x <= rect.x1:
                                    vertical_xs.add((p1.x, stroke_color, w))
                                    
                        # Reconstruct horizontal grid lines
                        elif abs(p1.y - p2.y) < 0.1:
                            if min(p1.x, p2.x) <= rect.x0 and max(p1.x, p2.x) >= rect.x1:
                                if rect.y0 <= p1.y <= rect.y1:
                                    horizontal_ys.add((p1.y, stroke_color, w))
            
            # Draw back the vertical grid lines perfectly on top of the covered area
            for x, col, w in vertical_xs:
                page.draw_line(fitz.Point(x, rect.y0), fitz.Point(x, rect.y1), 
                               color=col if col else grid_color, width=w, overlay=True)
                
            # Draw back the horizontal grid lines perfectly on top of the covered area
            for y, col, w in horizontal_ys:
                page.draw_line(fitz.Point(rect.x0, y), fitz.Point(rect.x1, y), 
                               color=col if col else grid_color, width=w, overlay=True)
                
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
                               
