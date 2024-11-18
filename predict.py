import requests
import json
from metaflow import Flow
import numpy as np
import os
import typer
from tabulate import tabulate

#This script is used to send a track name to a TensorFlow model deployed with TensorFlow Serving and retrieve recommendations based on that track.
def main(track: str = "Tom Petty|||Free Fallin'"):
    data = json.dumps({"instances": [track]})
    json_response = requests.post(
        f"http://localhost:8501/v1/models/{os.environ['TF_MODEL_NAME']}:predict",
        data=data,
        headers={"content-type": "application/json"},
    )
    predictions = json.loads(json_response.text)["predictions"]
    print(f"\nShowing predictions for track: {track}\n")
    print(tabulate(predictions[0]))  # assumes one prediction at a time


if __name__ == "__main__":
    typer.run(main)
