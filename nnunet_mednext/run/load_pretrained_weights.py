#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
import warnings

def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")


def load_pretrained_weights_notstrict(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    # for key, _ in model_dict.items():
    #     if ('conv_blocks' in key):
    #         if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
    #             continue
    #         else:
    #             ok = False
    #             break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        # if verbose:
        print("Below is the list of overlapping blocks in pretrained model and original architecture:")
        for key, _ in pretrained_dict.items():
            print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")


def load_pretrained_weights_resampling(network, fname, verbose=False):

    print("################### Resampled Loading pretrained weights from file ", fname, '###################')
    
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match. # Fabian wrote this.
    new_state_dict = {}
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict
    model_dict = network.state_dict()
    
    for k in model_dict.keys():
        # print(k, model_dict[k].shape, pretrained_dict[k].shape)

        if k in model_dict.keys() and k in pretrained_dict.keys():  # Common keys
            if 'bias' in k or 'norm' in k or 'dummy' in k:
                print(f"Key {k} loaded unchanged.")
                model_dict[k] = pretrained_dict[k]
            else:
                inc1, outc1, *spatial_dims1 = model_dict[k].shape
                inc2, outc2, *spatial_dims2 = pretrained_dict[k].shape
                print(inc1, outc1, spatial_dims1, inc2, outc2, spatial_dims2)

                assert inc1==inc2 # Please use equal in_channels in all layers for resizing pretrainer
                assert outc1 == outc2 # Please use equal out_channels in all layers for resizing pretrainer
                
                if spatial_dims1 == spatial_dims2:
                    model_dict[k] = pretrained_dict[k]
                    print(f"Key {k} loaded.")
                else:
                    if len(spatial_dims1)==3:
                        model_dict[k] = torch.nn.functional.interpolate(
                                                pretrained_dict[k], size=spatial_dims1,
                                                mode='trilinear'
                                                )
                        print(f"Key {k} interpolated trilinearly from {spatial_dims2}->{spatial_dims1} and loaded.")
                    elif len(spatial_dims1)==2:
                        model_dict[k] = torch.nn.functional.interpolate(
                                                pretrained_dict[k], size=spatial_dims1,
                                                mode='bilinear'
                                                )
                        print(f"Key {k} interpolated bilinearly from {spatial_dims2}->{spatial_dims1} and loaded.")
                    else:
                        raise TypeError('UpKern only supports 2D and 3D shapes.')
        else:   # Keys which are not shared
            warnings.warn(f"Key {k} in current_model:{k in model_dict.keys()} and pretrained_model:{k in pretrained_dict.keys()} and will not be loaded.")

    network.load_state_dict(model_dict)
    print("######## Weight Loading DONE ############")


def load_pretrained_weights_fusing(network, fname, verbose=False):

    print("################### Resampled Loading pretrained weights from file ", fname, '###################')
    
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match. # Fabian wrote this.
    # new_state_dict = {}
    # for k, value in pretrained_dict.items():
    #     key = k
    #     # remove module. prefix from DDP models
    #     if key.startswith('module.'):
    #         key = key[7:]
    #     new_state_dict[key] = value

    # pretrained_dict = new_state_dict
    model_dict = network.state_dict()
    
    for k in model_dict.keys():
        # print(k, model_dict[k].shape, pretrained_dict[k].shape)

        if k in model_dict.keys() and k in pretrained_dict.keys():  # Common keys
            if 'bias' in k or 'norm' in k:
                print(f"Key {k} added")
                model_dict[k] += pretrained_dict[k].cuda()
            else:
                inc1, outc1, *spatial_dims1 = model_dict[k].shape
                inc2, outc2, *spatial_dims2 = pretrained_dict[k].shape
                print(inc1, outc1, spatial_dims1, inc2, outc2, spatial_dims2)

                assert inc1==inc2 # Please use equal in_channels in all layers for resizing pretrainer
                assert outc1 == outc2 # Please use equal out_channels in all layers for resizing pretrainer
                
                if spatial_dims1 == spatial_dims2:
                    model_dict[k] += pretrained_dict[k].cuda()
                    print(f"Key {k} added.")
                else:
                    spatial_dims_diff = [spatial_dims1[i] - spatial_dims2[i] for i in range(len(spatial_dims1))]
                    pads = (spatial_dims_diff[0]//2, spatial_dims_diff[0]//2,
                            spatial_dims_diff[1]//2, spatial_dims_diff[1]//2,
                            spatial_dims_diff[2]//2, spatial_dims_diff[2]//2
                            )
                    model_dict[k] += torch.nn.functional.pad(pretrained_dict[k].cuda(),
                                            pads, mode='constant',
                                            value=0
                                            )
                    print(f"Key {k} added after padding and loaded.")
        else:   # Keys which are not shared
            warnings.warn(f"Key {k} in current_model:{k in model_dict.keys()} and pretrained_model:{k in pretrained_dict.keys()} and will not be loaded.")

    network.load_state_dict(model_dict)
    print("######## Weight Loading DONE ############")


if __name__ == "__main__":
    from nnunet_mednext.network_architecture.custom_modules.custom_networks.UNeXt.UNext \
    import UNeXt_Prototype as UNeXt
    model = UNeXt(
            in_channels = 1, 
            n_channels = 30,
            n_classes = 14, 
            exp_r=[4,4,4,4,4,4,4,4,4],         # Expansion ratio as in Swin Transformers
            kernel_size=7,                     # Ofcourse can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

    load_pretrained_weights_resampling(model, 'somewhere/PythonProjects/model_final_checkpoint.model')