from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import eV2L
import io
from PIL import Image
import asyncio

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # base64 string

# Load the model once
try:
    model = eV2L.load_model()
except FileNotFoundError:
    raise Exception("Model file not found")
except MemoryError:
    raise Exception("Not enough memory to load model")
except Exception as e:
    raise Exception(f"Unexpected error loading model: {e}")

# cors 이슈
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_api():
    return "This is the flower classifier API"

async def async_predict(model, image):
    # 동기 함수를 비동기적으로 실행
    try:
        result = await asyncio.to_thread(eV2L.predict, model, image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/")
async def predict_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 비동기 래퍼 함수를 통해 예측 실행
        raw_result = await async_predict(model, image)
        
        if raw_result is None:
            raise HTTPException(status_code=500, detail="Prediction returned None")
            
        if isinstance(raw_result, (list, tuple)):
            result = {"prediction": raw_result}
        elif not isinstance(raw_result, dict):
            result = {"prediction": str(raw_result)}
        else:
            result = raw_result
            
        return result
            
    except HTTPException as e:
        raise e
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Run the server
if __name__ == "__main__":
    uvicorn.run("server:app",
            reload= True,   # Reload the server when code changes
            host="127.0.0.1",   # Listen on localhost 
            port=5000,   # Listen on port 5000 
            log_level="info"   # Log level
            )