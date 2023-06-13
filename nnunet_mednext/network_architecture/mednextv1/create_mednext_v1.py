from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt

def create_mednextv1_small(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=2,                         
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2]
    )


def create_mednextv1_base(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2]
    )


def create_mednextv1_medium(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block'
    )


def create_mednextv1_large(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[3,4,8,8,8,8,8,4,3],                          
        kernel_size=kernel_size,                     
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,8,8,8,8,8,4,3],
        checkpoint_style = 'outside_block'
    )


def create_mednext_v1(num_input_channels, num_classes, model_id, kernel_size=3,
                      deep_supervision=False):

    model_dict = {
        'S': create_mednextv1_small,
        'B': create_mednextv1_base,
        'M': create_mednextv1_medium,
        'L': create_mednextv1_large,
        }
    
    return model_dict[model_id](
        num_input_channels, num_classes, kernel_size, deep_supervision
        )


if __name__ == "__main__":

    model = create_mednextv1_large(1, 3, 3, False)
    print(model)