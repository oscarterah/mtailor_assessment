import os
from PIL import Image
import numpy as np
from model import OnnxImageClassifier, Preprocessor
import requests
import argparse

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
    
def test_docker(image_path):
    with open(image_path, "rb") as f:
        files = {"model_bytes_str": ("test.jpg", f, "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)

    result = response.json()
    print(result)

    if result == 0:
        return True
    else:
        return False
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run tests for image classification.")
    parser.add_argument(
        "--docker",
        action="store_true", # Makes it a flag, True if present, False otherwise
        help="Run the Docker endpoint test."
    )
    args = parser.parse_args()

    results = {}
    all_tests_passed = True

    if args.docker:
        # If --docker is specified, only run the Docker test
        print("Running Docker endpoint test only as per --docker flag.")
        docker_test_success = test_docker(IMAGE_PATH)
        results["Docker Endpoint Test"] = docker_test_success
        if not docker_test_success:
            all_tests_passed = False
    else:
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