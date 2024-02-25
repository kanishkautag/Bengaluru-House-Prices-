import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import pickle
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model and columns
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open('columns.json', 'r') as json_file:
    columns_data = json.load(json_file)
    all_columns = columns_data.get("data_columns", [])


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


@app.post("/predict")
def predict_price(item: PriceInput):
    # Check if the location is in the columns
    if item.location not in all_columns:
        return {"error": "Invalid location"}

    # Convert location to dummy variables
    loc_index = all_columns.index(item.location)
    input_features = np.zeros(len(all_columns))
    input_features[0] = item.sqft
    input_features[1] = item.bath
    input_features[2] = item.bhk
    input_features[loc_index] = 1

    # Make the prediction
    predicted_price = model.predict([input_features])[0]

    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
