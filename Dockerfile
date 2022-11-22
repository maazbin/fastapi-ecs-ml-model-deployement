FROM python:3.8


RUN pip3 install --upgrade pip

# installing git

# RUN apt-get update && apt-get install -y git

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# RUN python -m pip install git+https://github.com/maazbin/keras-image-helper.git

# copy tensorflowlite model
COPY clothing-model-v4.tflite clothing-model-v4.tflite

COPY main.py main.py


CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]