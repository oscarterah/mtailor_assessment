import torch
import torch.nn.functional as F
import torch.onnx
import onnx
from pytorch_model import Classifier
import os

class PreprocessingModel(torch.nn.Module):
    def __init__(self, model: Classifier):
        super().__init__()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def forward(self, x):
        """this matches what was required in delivarables for preprocessing but does not use the
            example in pytorch_model.py because it causes torch __round__ errors
        """
        x = x / 255.0
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)
    

def convert_to_onnx(pytorch_model_path, onnx_model_path):

    """This link shows how to export a pytorch model with additional steps that are similar to preprocessing
        
        https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

    """


    model = Classifier()

    if torch.cuda.is_available():
        state_dict = torch.load(pytorch_model_path)
    else:
        state_dict = torch.load(pytorch_model_path, map_location='cpu')
    
    if 'state_dict' in state_dict:
        weights = state_dict['state_dict']
    elif 'model' in state_dict:
        weights = state_dict['model']
    else:
        weights = state_dict

    
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
    
    #this check ensures that the weight that are used are loaded from the actual model file
    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys[:5]}...")
    
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Model output shape: {test_output.shape}")
    
    wrapped_model = PreprocessingModel(model)
    print("Converting to ONNX format with preprocessing...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Successfully converted model to ONNX: {onnx_model_path}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model file size: {os.path.getsize(onnx_model_path) / (1024*1024):.2f} MB")
    
    return True

if __name__ == "__main__":
    pytorch_weights_path = "pytorch_model_weights.pth"
    onnx_model_path = "model.onnx"
    success = convert_to_onnx(pytorch_weights_path, onnx_model_path)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        exit(1)
