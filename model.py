import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Tuple, List


#delivarable
class Preprocessor:
    def __init__(self, input_shape=(224, 224)):
        self.input_shape = input_shape
        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        self.std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    #delivarable
    def preprocess(self, image_path:str):
        image = Image.open(image_path).convert('RGB')
        array = np.array(image, dtype=np.float32)
        #convert HWC to CHW and add batch dimension (1, C,H,W) creating a torch tensor format
        tensor = np.transpose(array, (2, 0, 1))[None, :, :, :]
        return tensor
    
    
#delivarable
class OnnxImageClassifier:
    def __init__(self, model_path):
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor):
        #preprocessing done in model
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = outputs[0][0]
        predicted_class_id = int(np.argmax(predictions))
        return predicted_class_id
