import torch
import mxnet as mx
import numpy as np
from gluon2pytorch import gluon2pytorch

import gluoncv
def check_error(gluon_output, pytorch_output, epsilon=1e-4):
    pytorch_output = pytorch_output.data.numpy()
    gluon_output = gluon_output.asnumpy()

    error = np.max(pytorch_output - gluon_output)
    print('Error:', error)

    assert error < epsilon
    return error

model_name = "resnet18_v1b_0.89"
model_name = "resnet50_v1d_0.86"
model_name = "resnet50_v1d_0.48"
model_name = "resnet50_v1d_0.37"
# model_name = "resnet50_v1d_0.11"
# model_name = "resnet101_v1d_0.76"
# model_name = "resnet101_v1d_0.73"

modelFile = {
    'resnet18_v1b_0.89': "resnet18_v1b_89",
    'resnet50_v1d_0.86': "resnet50_v1d_86",
    'resnet50_v1d_0.48': "resnet50_v1d_48",
    'resnet50_v1d_0.37': "resnet50_v1d_37",
    'resnet50_v1d_0.11': "resnet50_v1d_11",
    'resnet101_v1d_0.76': "resnet101_v1d_76",
    'resnet101_v1d_0.73': "resnet101_v1d_73",
}


model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# model = gluoncv.model_zoo.resnet50_v1d_11(pretrained=True)
print(model)


# Make sure it's hybrid and initialized
model.hybridize()


pytorch_model = gluon2pytorch(model, [(1, 3, 224, 224)], dst_dir="./", pytorch_module_name=modelFile[model_name] ,keep_names = True)
pytorch_model.eval()
input_np = np.random.uniform(-1, 1, (1, 3, 224, 224))

gluon_output = model(mx.nd.array(input_np))
pytorch_output = pytorch_model(torch.FloatTensor(input_np))
check_error(gluon_output, pytorch_output)
