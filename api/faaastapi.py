import uvicorn
import json
import logging
from fastapi import FastAPI, Request
from inference import get_recommendations

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello! You app is running.."}

@app.get("/predict")
async def get_preds(request: Request):
    id = request.query_params['id']
    response = get_recommendations(int(id))
    converted_response = {str(k):float(v) for k, v in response.items()}
    return json.dumps(converted_response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)