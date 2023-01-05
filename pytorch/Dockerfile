FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime 


RUN apt-get update && apt-get upgrade -y 

# libopencv-dev のためにtimezone設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# opencvのためにinstall
# https://qiita.com/narista/items/a3d7d26ae50d54c2553a
RUN apt-get install -y libopencv-dev 

RUN pip install fastapi uvicorn python-multipart opencv-python