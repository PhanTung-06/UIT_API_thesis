from utils.detector import Detector
from detectron2.data.detection_utils import read_image
from distutils.command.config import config
import torch
import numpy as np
import os
import glob

import redis
import json
import time
import sys
import base64
from PIL import Image
import io
import struct

# Connect to Redis server
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

model = Detector('./configs/ABINet/VinText.yaml', './model_0059999.pth')

CTLABELS = [' ',"'",'^', '\\', '}', 'ỵ','Ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
voc_size = 230
def decode(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < voc_size:
            s += CTLABELS[c]
    return s

def detect_process():
    while True:
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()
        
        imageIDs = []
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            img = base64.b64decode(q["image"])
            img = Image.open(io.BytesIO(img))
            image = np.array(img.convert('RGB'))[:, :, ::-1]
           
            print(image.shape)
           
        
            # # Check to see if the batch list is None
            # if batch is None:
            #     batch = image
            
            # Otherwise, stack the data
            # else:
            #     batch = np.vstack([batch, image])

            # Update the list of image IDs
            imageIDs.append(q["id"])
        
        # Check to see if we need to process the batch
        if len(imageIDs) > 0:
            # Detect the batch
            for imageID in imageIDs:
                print("* image size: {}".format(image.shape))
                pred, vis = model.predict(image)
               
                vis.save("output.jpg")
                with open("output.jpg", "rb") as image_file:
                    img = image_file.read()
                image =io.BytesIO(img)
                img_str = base64.b64encode(image.getvalue()).decode('utf-8')
                output = []
                r = {"output_img":img_str}
                output.append(r)
                db.set(imageID, json.dumps(output))
        # Sleep for a small amount
        time.sleep(float(os.environ.get("SERVER_SLEEP")))

if __name__ == "__main__":
    detect_process()