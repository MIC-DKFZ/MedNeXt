import pytest
import torch
from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1

class Test_MedNeXt_archs:

    @pytest.mark.parametrize("model_size, kernel_size", [
    ('S', 3),
    ('B', 3),
    ('M', 3),
    ('L', 3),
    ('S', 5),
    ('B', 5),
    ('M', 5),
    ('L', 5),
    ])
    def test_init_and_forward(self, model_size, kernel_size):
        m = create_mednext_v1(2, 4, model_size, kernel_size).cuda()
        input = torch.zeros((1,2,128,128,128), requires_grad=False).cuda()
        with torch.no_grad():
            output = m(input)
        del m
        inp_shape = input.shape
        assert output[0].shape == (inp_shape[0], 4, *inp_shape[2:])
