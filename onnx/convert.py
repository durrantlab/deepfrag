
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.onnx

import onnx
import onnxruntime

import numpy as np

from leadopt.models.voxel import VoxelFingerprintNet2b


# model params
MODEL_PATH = './200618_m150_simple.pt'
REC_CHANNELS = 4
LIG_CHANNELS = 3

# load pytorch model
m = VoxelFingerprintNet2b(
    in_channels=REC_CHANNELS + LIG_CHANNELS,
    output_size=2048,
    blocks=[32,64],
    fc=[2048],
    pad=False
)
m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
m = m.eval()
print(m)

# create fake input/output for conversion
x = torch.zeros(1,7,24,24,24)
torch_out = m(x)
print('torch output: ', torch_out)

# export as onnx model
torch.onnx.export(
    m,
    x,
    './200618_m150_simple.onnx',
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# check onnx model
onnx_model = onnx.load('./200618_m150_simple.onnx')
onnx.checker.check_model(onnx_model)

# test onnx model output
ort_session = onnxruntime.InferenceSession('./200618_m150_simple.onnx')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

print('onnx output: ', ort_outs[0])

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print('done')
