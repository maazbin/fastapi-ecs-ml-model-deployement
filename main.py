# from enum import Enum

#fast api modules
# from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile


#tensorflow lite module
import tflite_runtime.interpreter as tflite

#   other image processing
from PIL import Image
from keras_image_helper import create_preprocessor
# other modules
from io import BytesIO





app = FastAPI()



def preprocess(file):

    # read image from  requeset  
    img = file.file.read()
    
    #   mimicing os file support behaviour for bytes stream 
    stream = BytesIO(img)
    
    # stream to image
    img = Image.open(stream)

    # creating preprocessor for Xception
    preprocessor = create_preprocessor('xception', target_size=(299, 299))  
    X = preprocessor.convert_to_tensor(img)


    return X



def model(X):

    # Creates the TF Lite interpreter (load model)
    interpreter = tflite.Interpreter(model_path='clothing-model-v4.tflite') 
    
    # Initializes the interpreter with the model
    interpreter.allocate_tensors() 
    
    # Gets the input: the part of the network that takes in the array X
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index'] 
    
    # Gets the output: the part of the network with final predictions
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index'] 
    
    # Puts X into the input
    interpreter.set_tensor(input_index, X)
    
    # Runs the model to get predictions
    interpreter.invoke()

    # Gets the predictions from the output
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

    # combine labels with prediction results
    results = dict(zip(labels, preds[0]))
    
    
    # Maximum value of dictionary (highest probability)
    max_value = max(results, key=results.get)

    return max_value



@app.post("/")
async def get_prediction_root(file: UploadFile = File(...)):
    prep = preprocess(file)
    preds = model(prep)
    # preds = prediction(interpreter)
    decode_pred = decode(preds)
    return {"prediction" : decode_pred}


#health check for alb

@app.get("/health")
async def get_health_check():
    
    return {"health status": "Good !"}


#health check for alb

@app.get("/labels")
async def get_lables():
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
    return {"labels": labels}