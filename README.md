# Deploying a Classification Neural Network on Serverless GPU platform of Cerebrium

## Requirements

- Python 3.10+
- Docker
- Pytorch
- Cerebrium account (30 USD free credits)
- Git


### 1. Download Model Weights

Manually downloading the model weights:

```bash
wget -O pytorch_model_weights.pth "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0"
```
### 2. Create and activate a virtual environment 
Assuming use of linux or mac but tested on linux

```bash
python3.10 -m venv mtailorvenv3.10
source mtailorvenv3.10/bin/activate
```

### 3. Install requirements

```bash
pip install -r requiremnents.txt
```

### 4. Run convert_to_onnx.py to convert model
This converts the downloaded pytorch model to onnx but with preprocessing in the onnx output file

```bash
python convert_to_onnx.py
```

### 5. Runing Tests
This test the model.onnx generated if it opens well, tests the preprocessor class

```bash
python test.py
```

### 6. Build Docker Image

```bash
docker build . --file Dockerfile --tag imagenet-classifier:ci
```

### 7. Test Docker Image
```bash
python test.py -- docker
```