# from enum import Enum

#fast api modules
# from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile


#tensorflow lite module
import tflite_runtime.interpreter as tflite

#   other image processing
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input

# other modules
import numpy as np 
from io import BytesIO





app = FastAPI()



def preprocess(file):

    img = file.file.read()
    
    #   mimicing os file support behaviour for bytes stream 
    stream = BytesIO(img)
    

    img = Image.open(stream)
    
    #to np array
    x = np.array(img)
    # adding one more dim
    X = np.array([x])

    # preprocessor for Xception
    X = preprocess_input(X)

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
    
    
    # Maximum value of dictionary (highest probability)
    max_value = max(results, key=results.get)

    return max_value


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/")
async def get_prediction_root(file: UploadFile = File(...)):
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
