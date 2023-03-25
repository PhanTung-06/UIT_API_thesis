from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from starlette.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import json
import os
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw
import redis
import shutil
import base64
import struct

import time 

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))
CLIENT_MAX_TRIES = 100000


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

@app.post("/predict")
async def predict(request: Request, img_file: UploadFile=File(...)):
    data = {"success": False}

    if request.method == "POST":
        # image = io.BytesIO(img_file)
        # img = Image.open(img_file)
        image = img_file.file.read()
        image = io.BytesIO(image)
        image_str = base64.b64encode(image.getvalue()).decode('utf-8')
        k = str(uuid.uuid4())
        d = {"id": k, "image": image_str}
        # print(d)
        # time.sleep(100)
        db.rpush(os.environ.get("IMAGE_QUEUE"), json.dumps(d))

        # Keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1  

            output = db.get(k)
            if output is not None:
                # Add the output predictions to our data dictionary so we can return it to the client
                output = output.decode("utf-8")
                data["img_result"] = json.loads(output)

                # Delete the result from the database and break from the polling loop
                db.delete(k)
                break

        # Sleep for a small amount to give the model a chance to classify the input image
        time.sleep(float(os.environ.get("CLIENT_SLEEP")))
        # Indicate that the request was a success
        data["success"] = True
    else:
        raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    # print(data["img_result"])
    for i in data["img_result"]:
        img_result = i["output_img"]

   
    return templates.TemplateResponse("result.html", {
        "request": request, "img_result": img_result
    })

