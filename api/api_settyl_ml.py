import os
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

this_dir = os.path.dirname(__file__)
utils_dir = os.path.join(this_dir, '..')
sys.path.append(utils_dir)

from pre_processing.pre_process import PRE_PROCESS

app = FastAPI()


class InputData(BaseModel):
    external_status: str


@app.post("/settyl-predict/")
def predict_internal_status(input_data: InputData):
    processed_data = input_data.external_status
    predicted_value, f1score, accuracy, precision, recall = PRE_PROCESS().predict_internal_status([processed_data])
    return {"internal_status": predicted_value, "f1score": f1score, "accuracy": accuracy, "precision": precision,
            "recall": recall}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
