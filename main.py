from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()

# Enable CORS for your Vercel frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/remove-watermark/")
async def remove_watermark(image: UploadFile = File(...)):
    # 1. Read the incoming image file bytes
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    
    # 2. Create a completely black mask matching the image dimensions
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 3. Target ONLY the specific bottom-right corner zone where NotebookLM is located
    # This prevents the algorithm from altering any main diagrams or central text structures
    crop_start_y = int(h - 60)   # 60 pixels from the bottom
    crop_start_x = int(w - 220)  # 220 pixels from the right
    
    watermark_zone = img[crop_start_y:h, crop_start_x:w]
    
    # 4. Convert the cropped zone to grayscale to analyze pixel intensity
    gray_zone = cv2.cvtColor(watermark_zone, cv2.COLOR_BGR2GRAY)
    
    # 5. Isolate only the dark text pixels of the watermark using binary inversion thresholding
    # This avoids selecting the background color and extracts just the letter shapes
    _, text_mask = cv2.threshold(gray_zone, 110, 255, cv2.THRESH_BINARY_INV)
    
    # 6. Apply the precise text mask back into the global mask coordinate system
    mask[crop_start_y:h, crop_start_x:w] = text_mask
    
    # 7. Run Inpainting with a very small radius (3)
    # Instead of drawing a solid box, it seamlessly refills only the empty letter paths
    # using the immediate surrounding background color gradient
    cleaned_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # 8. Encode the processed clean matrix into memory bytes and stream it back
    _, encoded_img = cv2.imencode('.png', cleaned_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
  
