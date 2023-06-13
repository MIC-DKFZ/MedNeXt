import torch
import torch.nn as nn
import warnings

def upkern_load_weights(network:nn.Module, pretrained_net:nn.Module):
    pretrained_dict = pretrained_net.state_dict()
    model_dict = network.state_dict()
    
    for k in model_dict.keys():
        # print(k, model_dict[k].shape, pretrained_dict[k].shape)

        if k in model_dict.keys() and k in pretrained_dict.keys():  # Common keys
            if 'bias' in k or 'norm' in k or 'dummy' in k:          # bias, norm and dummy layers
                print(f"Key {k} loaded unchanged.")
                model_dict[k] = pretrained_dict[k]
            else:                                                   # Conv / linear layers
                inc1, outc1, *spatial_dims1 = model_dict[k].shape
                inc2, outc2, *spatial_dims2 = pretrained_dict[k].shape
                print(inc1, outc1, spatial_dims1, inc2, outc2, spatial_dims2)

                assert inc1==inc2 # Please use equal in_channels in all layers for resizing pretrainer
                assert outc1 == outc2 # Please use equal out_channels in all layers for resizing pretrainer
                
                if spatial_dims1 == spatial_dims2:
                    model_dict[k] = pretrained_dict[k]
                    print(f"Key {k} loaded.")
                else:
                    model_dict[k] = torch.nn.functional.interpolate(
                                            pretrained_dict[k], size=spatial_dims1,
                                            mode='trilinear'
                                            )
                    print(f"Key {k} interpolated trilinearly from {spatial_dims2}->{spatial_dims1} and loaded.")
        else:   # Keys which are not shared
            warnings.warn(f"Key {k} in current_model:{k in model_dict.keys()} and pretrained_model:{k in pretrained_dict.keys()} and will not be loaded.")

    network.load_state_dict(model_dict)
    print("######## Weight Loading DONE ############")
    return network