from datetime import datetime
from typing import Union, Optional, List

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, File, UploadFile
from classes import CLASSESS
import io

import torch
import torchvision
# import torchvision.transforms as transforms
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import io
import numpy as np
import cv2

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



# @app.get("/predict")
# def test_return_json():
#     return JSONResponse(content={"Hello": "World"})



# https://qiita.com/sumita_v09/items/f1ada937ec64729b6c63
def read_image(bin_data, size=(224, 224)):
    # cv2だからBytesIOの処理ごちゃってるだけでPILならImage.openでもよい? -> 結局PILでやってる
    # https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8

    
    
    file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, size)

    return img

imsize = 224
tfs = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 岡村さんからもらったコード
# def save_upload_file_tmp(upload_file: UploadFile) -> Path:
#     try:
#         suffix = Path(upload_file.filename).suffix
#         with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             shutil.copyfileobj(upload_file.file, tmp)
#         tmp_path = Path(tmp.name)
#     except Exception as e:
#     finally:
#         upload_file.file.close()
#     return tmp_path
# @app.post("/predict/")
# def predict(file: UploadFile = File(...)):
#     filepath = save_upload_file_tmp(file)

# https://towardsdatascience.com/image-classification-api-with-tensorflow-and-fastapi-fc85dc6d39e8
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict")
async def image_recognition(file:UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = await file.read() # UploadFileのread()はbytes形式 # https://fastapi.tiangolo.com/tutorial/request-files/#:~:text=%E3%83%91%E3%82%B9%E6%93%8D%E4%BD%9C-,%E9%96%A2%E6%95%B0,-%E5%86%85%E3%81%A7%E3%81%AF%E3%80%81%E6%AC%A1
    image = io.BytesIO(image) # bytesioにする
    image = Image.open(image)# image = read_image(image) # cv2 array
    image = tfs(image).float() # transformed
    image = image.unsqueeze(0).cuda() # # https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411

    # https://note.nkmk.me/python-pytorch-pretrained-models-image-classification/
    model  = models.vgg16(pretrained=True).cuda()
    model.eval()
    with torch.no_grad():
        idx = int(torch.argmax(model(image)[0]))
        return {"Class":CLASSESS[idx]}


    # return {'filename': file.filename}
    """画像認識API

    Keyword Arguments:
        files {List[UploadFile]} -- アップロードされたファイル情報 (default: {File(...)})

    Returns:
        dict -- 推論結果
    """
    # pass
    # print(files[0])
    # bin_data = io.BytesIO(files[0].file.read())
    # image = read_image(bin_data)
    # image = tfs(image).float()
    # image = image.unsqueeze(0).cuda() 
    

    # with torch.no_grad():
    #     idx = torch.argmax(model(image)[0])
    #     return {"Class",CLASSESS[idx]}



# bytes base64 ByteIOについて
# https://gist.github.com/kapib/bca21a8a95ee0f74f313929f56beec35