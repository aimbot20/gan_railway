from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from model import stylize_image

app = FastAPI()

@app.post("/stylize")
async def stylize(file: UploadFile = File(...)):
    input_image = Image.open(file.file).convert("RGB")
    output_image = stylize_image(input_image)

    # Save to bytes
    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpg")
