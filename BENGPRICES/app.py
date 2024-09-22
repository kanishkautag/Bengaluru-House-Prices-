import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import pickle
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model and columns
try:
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

try:
    with open('columns.json', 'r') as json_file:
        columns_data = json.load(json_file)
        all_columns = columns_data.get("data_columns", [])
    logger.info(f"Columns loaded successfully. Total columns: {len(all_columns)}")
except Exception as e:
    logger.error(f"Error loading columns: {str(e)}")
    all_columns = []

def get_locations(query: str = Query(None)):
    if query:
        filtered_locations = [loc for loc in all_columns if query.lower() in loc.lower()]
        return JSONResponse(content={"locations": filtered_locations})
    else:
        return {"locations": all_columns}

# Define the input data model using Pydantic
class PriceInput(BaseModel):
    location: str
    sqft: float
    bath: int
    bhk: int

@app.get("/locations")
def fetch_locations(query: str = Query(None)):
    return get_locations(query)

@app.get("/predict", response_class=HTMLResponse)
async def serve_predict_page():
    try:
        with open("static/index.html", "r") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return HTMLResponse(content="<h1>Error loading page</h1>", status_code=500)

@app.post("/predict_price")
def predict_price(item: PriceInput):
    logger.debug(f"Received input: {item}")
    
    if not model:
        logger.error("Model not loaded")
        return {"error": "Model not available"}
    
    # Check if the location is in the columns
    if item.location not in all_columns:
        logger.warning(f"Invalid location: {item.location}")
        return {"error": "Invalid location"}

    try:
        # Convert location to dummy variables
        loc_index = all_columns.index(item.location)
        input_features = np.zeros(len(all_columns))
        input_features[0] = item.sqft
        input_features[1] = item.bath
        input_features[2] = item.bhk
        input_features[loc_index] = 1

        # Make the prediction
        predicted_price = model.predict([input_features])[0]
        logger.debug(f"Predicted price: {predicted_price}")
        
        return {"predicted_price": predicted_price}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": "Error during prediction"}

@app.get("/")
async def root():
    return {"message": "Welcome to the house price prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
