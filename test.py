import os
from PIL import Image
import numpy as np
from model import OnnxImageClassifier, Preprocessor

IMAGE_PATH = "n01440764_tench.jpeg"
ONNX_MODEL_PATH = "model.onnx" 



def test_preprocessor_preprocess_method():
    try:
        print("\n Testing Preprocessor.preprocess() method ")
        preprocessor = Preprocessor()
        tensor = preprocessor.preprocess(IMAGE_PATH)
        assert isinstance(tensor, np.ndarray), "Output is not a numpy array"
        assert tensor.ndim == 4, f"Output tensor should be 4D, got {tensor.ndim}D"
        assert tensor.shape[1] == 3, f"Output tensor should have 3 channels, got {tensor.shape[1]}"
        assert tensor.dtype == np.float32, f"Output tensor dtype should be float32, got {tensor.dtype}"
        print(f"Preprocessor.preprocess() successful. Output shape: {tensor.shape}, dtype: {tensor.dtype}")
        return True
    except Exception as e:
        print(f"Error in Preprocessor.preprocess(): {e}")
        return False

def test_onnx_classifier_instantiation():
    try:
        classifier = OnnxImageClassifier(ONNX_MODEL_PATH)
        assert hasattr(classifier, 'session'), "Classifier missing 'session' attribute"
        assert hasattr(classifier, 'input_name'), "Classifier missing 'input_name' attribute"
        assert hasattr(classifier, 'output_name'), "Classifier missing 'output_name' attribute"
        print("OnnxImageClassifier instantiated successfully.")
        return True
    except Exception as e:
        print(f"Error instantiating OnnxImageClassifier: {e}")
        return False

def test_onnx_classifier_prediction():
        
    try:
        preprocessor = Preprocessor()
        input_tensor = preprocessor.preprocess(IMAGE_PATH)
        print(f"Using input tensor of shape: {input_tensor.shape} for ONNX model")
        classifier = OnnxImageClassifier(ONNX_MODEL_PATH)
        predicted_class_id = classifier.predict(input_tensor)
        assert isinstance(predicted_class_id, int), "Predicted class ID is not an integer"
        print(f"OnnxImageClassifier prediction successful. Predicted class ID: {predicted_class_id}")
        return True
    
    except Exception as e:
        print(f"Error during OnnxImageClassifier prediction: {e}")
        return False
    
if __name__ == "__main__":
 
    results = {}
    results["Preprocessor.preprocess()"] = test_preprocessor_preprocess_method()
    results["OnnxImageClassifier Instantiation"] = test_onnx_classifier_instantiation()
    results["OnnxImageClassifier Prediction"] = test_onnx_classifier_prediction()

    print("\n--- Test Summary ---")
    all_passed = True
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")