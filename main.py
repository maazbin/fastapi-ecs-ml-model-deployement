from enum import Enum

#fast api modules
# from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile


#tensorflow module
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# other modules
import numpy as np 
# from keras_image_helper import create_preprocessor
# import tflite_runtime.interpreter as tflite

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}




def preprocess(img):

    # fullname = "t-shirt.png"
    # img = load_img(fullname, target_size=(299, 299))
    X = np.array(img)
    X = preprocess_input(X)
    X = np.array([X])

    return X



def model(X):
    interpreter = tflite.Interpreter(model_path='clothing-model-v4.tflite') 
    interpreter.allocate_tensors() 
    

    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index'] 
    
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index'] 
    
    interpreter.set_tensor(input_index, X)
    
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index) 
    
    return preds


# def prediction(interpreter):
#     interpreter,output_index = interpreter
#     preds = interpreter.get_tensor(output_index) 
#     return preds
    

    
    
def decode(preds):
    labels = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
    ]
    results = dict(zip(labels, preds[0]))
    max_value = max(results, key=results.get)

    return max_value


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/")
async def get_prediction(file: UploadFile):
    prep = preprocess(file)
    preds = model(prep)
    # preds = prediction(interpreter)
    decode_pred = decode(preds)
    return {"prediction" : decode_pred}

# preprocessor = create_preprocessor('xception', target_size=(299, 299))


# interpreter = tflite.Interpreter(model_path='clothing-model-v4.tflite')
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# input_index = input_details[0]['index']

# output_details = interpreter.get_output_details()
# output_index = output_details[0]['index']


# def predict(X):
#     interpreter.set_tensor(input_index, X)
#     interpreter.invoke()

#     preds = interpreter.get_tensor(output_index)
#     return preds[0]




# def decode_predictions(pred):
#     result = {c: float(p) for c, p in zip(labels, pred)}
#     return result
