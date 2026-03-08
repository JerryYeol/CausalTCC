class Config(object):
    def __init__(self):
        # model configs
        # !!! 已修正: 根据你的数据 torch.Size([..., 19, 1000]) 进行修改
        self.input_channels = 128          
        
        # !!! 已添加: 根据你的数据 torch.Size([..., 19, 1000]) 进行添加
        # self.sequence_length = 1000       

        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 5
        self.dropout = 0.5
        
        # 这个值很可能需要修改，先随便设置一个，之后用我给你的脚本去计算正确的值
        # self.features_len = 1 
        self.features_len = 18
        # training configs
        self.num_epoch = 160

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10

