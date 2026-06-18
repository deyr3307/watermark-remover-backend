from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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
    return {"message": "Ultimate Vector-Stream PDF Cleaner is Running!"}

@app.post("/remove-watermark/")
async def remove_watermark(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    
    # Open the PDF natively as a pure vector document
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        # Decompress and merge all layout content streams into a single manageable layer
        page.clean_contents()
        
        for xref in page.get_contents():
            # Extract the raw uncompressed data stream of the page
            data = doc.xref_stream(xref)
            
            # METHOD 1: Target literal text structures e.g., (NotebookLM) -> (          )
            # Replaces the characters inside the rendering brackets with completely transparent spaces
            data = re.sub(
                b"\\(([^)]*?Notebook[^)]*?)\\)", 
                lambda m: b"(" + b" " * len(m.group(1)) + b")", 
                data, 
                flags=re.IGNORECASE
            )
            
            # METHOD 2: Target Hex-encoded text structures e.g., <4e6f7465626f6f6b4c4d>
            # 4e6f7465626f6f6b is "Notebook" in hex format. Replaces pairs with '20' (Hex for space)
            data = re.sub(
                b"<([0-9a-fA-F]*?4e6f7465626f6f6b[0-9a-fA-F]*?)>", 
                lambda m: b"<" + b"20" * (len(m.group(1)) // 2) + b">", 
                data, 
                flags=re.IGNORECASE
            )
            
            # Write the modified clean vector stream back into the PDF object dictionary
            doc.update_stream(xref, data)
            
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
    
