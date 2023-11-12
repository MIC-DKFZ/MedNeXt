import torch
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt    

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count_table

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyse(k):

    with torch.no_grad():
        print(f'MedNeXt-S kernel_size={k} analysis')
        network = MedNeXt(
                in_channels = 1, 
                n_channels = 32,
                n_classes = 13,
                # exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
                exp_r = 2,
                kernel_size=k,                     # Can test kernel_size
                deep_supervision=True,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                block_counts = [2,2,2,2,2,2,2,2,2],
                # block_counts = [3,4,8,8,8,8,8,4,3],
                checkpoint_style = None,
                grn=True
            ).cuda()
        
        print(count_parameters(network))

        # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
        x = torch.zeros((1,1,128,128,128), requires_grad=False).cuda()
        flops = FlopCountAnalysis(network, x)
        print(flops.total())


        print(f'\nMedNeXt-B kernel_size={k} analysis')
        network = MedNeXt(
                in_channels = 1, 
                n_channels = 32,
                n_classes = 13,
                exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
                # exp_r = 2,
                kernel_size=k,                     # Can test kernel_size
                deep_supervision=True,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                block_counts = [2,2,2,2,2,2,2,2,2],
                # block_counts = [3,4,8,8,8,8,8,4,3],
                checkpoint_style = None,
                grn=True
            ).cuda()
        
        print(count_parameters(network))

        # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
        x = torch.zeros((1,1,128,128,128), requires_grad=False).cuda()
        flops = FlopCountAnalysis(network, x)
        print(flops.total())


        print(f'\nMedNeXt-M kernel_size={k} analysis:')
        network = MedNeXt(
                in_channels = 1, 
                n_channels = 32,
                n_classes = 13,
                exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
                # exp_r = 2,
                kernel_size=k,                     # Can test kernel_size
                deep_supervision=True,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                # block_counts = [2,2,2,2,2,2,2,2,2],
                block_counts = [3,4,4,4,4,4,4,4,3],
                checkpoint_style = None,
                grn=True
            ).cuda()
        
        print(count_parameters(network))

        # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
        x = torch.zeros((1,1,128,128,128), requires_grad=False).cuda()
        flops = FlopCountAnalysis(network, x)
        print(flops.total())

        print(f'\nMedNeXt-L kernel_size={k} analysis:')
        network = MedNeXt(
            in_channels = 1, 
                n_channels = 32,
                n_classes = 13, 
                exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
                kernel_size=k,                     # Can test kernel_size
                deep_supervision=True,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                # block_counts = [6,6,6,6,4,2,2,2,2],
                block_counts = [3,4,8,8,8,8,8,4,3],
                # checkpoint_style = 'outside_block',
                grn=True
            ).cuda()
        
        print(count_parameters(network))

        # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
        x = torch.zeros((1,1,128,128,128), requires_grad=False).cuda()
        flops = FlopCountAnalysis(network, x)
        print(flops.total())
        
        print(f'\nMedNeXt-XL kernel_size={k} analysis:')
        network = MedNeXt(
            in_channels = 1, 
                n_channels = 32,
                n_classes = 13, 
                exp_r=[8,8,16,16,32,16,16,8,8],         # Expansion ratio as in Swin Transformers
                kernel_size=k,                     # Can test kernel_size
                deep_supervision=True,             # Can be used to test deep supervision
                do_res=True,                      # Can be used to individually test residual connection
                do_res_up_down = True,
                # block_counts = [6,6,6,6,4,2,2,2,2],
                block_counts = [8,8,16,16,32,16,16,8,8],
                # checkpoint_style = 'outside_block',
                grn=True
            ).cuda()
        
        print(count_parameters(network))

        # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
        x = torch.zeros((1,1,128,128,128), requires_grad=False).cuda()
        flops = FlopCountAnalysis(network, x)
        print(flops.total())


if __name__ == "__main__":
    for k in [3,5,7,9]:    
        analyse(k)