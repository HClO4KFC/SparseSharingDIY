import torch.nn

class GenDisparityHead(torch.nn.Module):
    def __init__(self, cv_task_arg, in_features, out_features=1):
        super(GenDisparityHead, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=256, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=cv_task_arg.scale_factor, mode=cv_task_arg.upsample_mode),
            torch.nn.Conv2d(in_channels=256, out_channels=out_features, kernel_size=1)  #输出深度图的通道数为1
        )

    def forward(self, x):
        depth_map = self.decoder(x)
        return depth_map


class GenRightHead(torch.nn.Module):
    def __init__(self, cv_task_arg, in_features, out_features):
        super(GenRightHead, self).__init__()
        self.decoder = torch.nn.Sequential(
            
        )

