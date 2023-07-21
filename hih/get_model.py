try : 
    from HIH import StackedHourGlass
except ModuleNotFoundError : 
    from hih.HIH import StackedHourGlass

class Default_Config():
    def __init__(self, **kwargs):
        self.input_size = 256
        self.heatmap_size = 64
        self.heatmap_method = "GAUSS"
        self.heatmap_sigma = 1.5
        self.pretrained = False
        self.inference_indice = -1
        self.mlp_r = 2
        self.per_stack_heatmap = 1 
        self.head_type = 'hih'
        self.data_type = 'WFLW'
        self.backbone = 'hourglass'
        self.heatmap_size = 64
        self.num_stack = 4
        self.num_layer = 4
        self.num_feature = 256
        self.offset_size = 4
        self.target_o = 384
        self.offset_method = "GAUSS"
        self.offset_sigma = 1.0
        self.criterion_heatmap = 'l2'
        self.criterion_offset = 'cls'
        self.loss_heatmap_weight = 1
        self.loss_offset_weight = 0.05
        self.num_landmarks = 98
        self.pretrained_path = ''
        self.refine_heatmap = False
        self.separate_landmark = False
        
        for key, value in kwargs.items():
            if not hasattr(self, key) : 
                raise KeyError(f'Invalid option : {key}')
            setattr(self, key, value)
        
        
def HIH_model(**kwargs) : 
    config = Default_Config(**kwargs)
    model = StackedHourGlass(config)
    return model 


if __name__ == '__main__' : 
    import torch 
    import pdb 
    model = HIH_model(input_size=512, heatmap_size=128, num_landmarks=16, offset_size=4).cuda()
    x = torch.randn(4,1,512,512).cuda()
    out_preds, out_offsets = model(x)
    print(out_preds.shape, out_offsets.shape) # (5,4,16,64,64), (5,1,16,8,8)
    pdb.set_trace()