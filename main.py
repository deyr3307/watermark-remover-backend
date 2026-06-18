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
    return {"message": "Grid-Clone Masterclass PDF Cleaner is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open PDF natively as a pure vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # STEP 1: Deep Core Stream Wiping (Removes text codes directly from the binary layer)
    for x in range(1, doc.xref_length()):
        try:
            stream = doc.xref_stream(x)
            if stream and (b"NotebookLM" in stream or b"4e6f7465626f6f6b4c4d" in stream or b"4E6F7465626F6F6B4C4D" in stream):
                stream = stream.replace(b"NotebookLM", b"          ")
                stream = stream.replace(b"4e6f7465626f6f6b4c4d", b"20202020202020202020")
                stream = stream.replace(b"4E6F7465626F6F6B4C4D", b"20202020202020202020")
                doc.update_stream(x, stream)
        except:
            pass

    # STEP 2: Page-Level Surgical Verification & Grid Clone Stamping
    for page in doc:
        page.clean_contents()
        
        # Look for any remaining layout traces of the watermark
        text_instances = page.search_for("NotebookLM")
        
        if text_instances:
            for rect in text_instances:
                # Calculate the exact height of the watermark box
                h_diff = rect.y1 - rect.y0
                
                # Dynamic Clone Source: Target the exact clean grid section right above the watermark
                # Shifting up vertically ensures the grid lines line up mathematically and perfectly
                source_rect = fitz.Rect(rect.x0, rect.y0 - h_diff - 12, rect.x1, rect.y0 - 12)
                
                # Capture the pristine background grid texture at ultra-high 450 DPI
                pix = page.get_pixmap(clip=source_rect, dpi=450)
                grid_bytes = pix.tobytes("png")
                
                # Stamp the clean grid texture exactly over the dirty watermark text area
                # This obliterates the text and naturally leaves the grid running through flawlessly
                page.insert_image(rect, stream=grid_bytes)
        else:
            # Absolute Fallback: Target the corner zone if font dictionary metadata is hidden
            p_width = page.rect.x1
            p_height = page.rect.y1
            fallback_rect = fitz.Rect(p_width - 150, p_height - 42, p_width - 15, p_height - 10)
            
            # Clone clean grid texture from 45 points above the fallback zone
            source_rect = fitz.Rect(p_width - 150, p_height - 87, p_width - 15, p_height - 55)
            pix = page.get_pixmap(clip=source_rect, dpi=450)
            grid_bytes = pix.tobytes("png")
            
            page.insert_image(fallback_rect, stream=grid_bytes)
            
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
            
