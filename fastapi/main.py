from datetime import datetime
from typing import Union, Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}




# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.get("/test_api")
def test_return_json():
    return JSONResponse(content={"Hello": "World"})



# class Item(BaseModel):
#     title: str
#     timestamp: datetime
#     description: Union[str, None] = None




# @app.put("/items/{id}")
# def update_item(id: str, item: Item):
#     json_compatible_item_data = jsonable_encoder(item)
#     return JSONResponse(content=json_compatible_item_data)



# コマンドからやる
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# と
# main.pyでやる
# if __name__ == '__main__':
#     uvicorn.run("main:app", port=8000, reload=True)
# がある
# https://qiita.com/alrar_yuri/items/8a3e6927ad9885f46a89