import numpy as np
import os
import omegaconf
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from datasets.dataLoader import SingleDataset
from model.mtlModel import build_backbone


class DepthEstimationHead(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1):
        super(DepthEstimationHead, self).__init__()

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        # 卷积层
        self.conv1 = nn.Conv2d(num_input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # 输出层
        self.output_conv = nn.Conv2d(64, num_output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 上采样
        x = self.upsample(x)

        # 卷积
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))

        # 输出层
        x = self.output_conv(x)

        return x


class DepthEstimationModel(torch.nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.backbone, out_channels = build_backbone('MobileNetV3Small')
        self.head = DepthEstimationHead(num_input_channels=out_channels, num_output_channels=1)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out


class ToFloat16Tensor(transforms.ToTensor):
    def __call__(self, pic):
        # 将 PIL 图像或 numpy 数组转换为 torch tensor
        tensor = super().__call__(pic)
        return tensor.to(torch.float16)


def main():
    iter_num = 200
    args = omegaconf.OmegaConf.load(os.path.join('yamls', 'default.yaml'))
    ds_trn = SingleDataset(
        dataset='cityscapes',
        path_pre=os.path.join('..', 'cvDatasets'),
        cv_task_arg=args.cv_tasks_args[0],
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='train',
        transform=transforms.Compose([ToFloat16Tensor()]),
        label_id_maps={})
    ds_val = SingleDataset(
        dataset='cityscapes',
        path_pre=os.path.join('..', 'cvDatasets'),
        cv_task_arg=args.cv_tasks_args[0],
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='val',
        transform=transforms.Compose([ToFloat16Tensor()]),
        label_id_maps={})
    trn_loader = DataLoader(
        dataset=ds_trn,
        shuffle=True,
        batch_size=1
    )
    val_loader = DataLoader(
        dataset=ds_val,
        shuffle=False,
        batch_size=1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthEstimationModel().half().to(device)
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.L1Loss()
    trn_loss_sav = []
    val_loss_sav = []
    for iter in range(iter_num):
        print(f'iter{iter}:', end='')
        model.train()
        trn_loss_iter = []
        for batch_idx, (batch_x, batch_std) in enumerate(trn_loader):
            length_div10 = len(trn_loader) // 10
            if batch_idx % length_div10 == length_div10 - 1:
                print('*', end='')
            batch_x, batch_std = batch_x.to(device), batch_std.to(device)
            optim.zero_grad()
            batch_y = model(batch_x)
            loss = criterion(batch_y, batch_std)
            trn_loss_iter.append(loss)
            loss.backward()
            trn_loss_iter.append(loss.detach().cpu())
            optim.step()
            del batch_x, batch_y, batch_std
        trn_loss_sav.append(np.avg(trn_loss_iter))
        model.eval()
        val_loss_iter = []
        print()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_std) in enumerate(val_loader):
                length_div10 = len(val_loader) // 10
                if batch_idx % length_div10 == length_div10 - 1:
                    print('*', end='')
                batch_y = model(batch_x)
                loss = criterion(batch_y, batch_std)
                batch_y.detach().cpu()
                batch_std.detach().cpu()
                batch_x.detach().cpu()
                val_loss_iter.append(loss.detach().cpu())
            val_loss_sav.append(np.avg(val_loss_iter))

    # 准备数据
    x1 = trn_loss_sav
    x2 = val_loss_sav
    y = range(len(trn_loss_sav))

    # 创建图形和子图
    plt.figure(figsize=(8, 6))

    # 绘制折线图
    plt.plot(x1, y, marker='o', linestyle='-')
    plt.plot(x2, y, marker='x', linestyle='.')

    # 添加标题和标签
    plt.title('Line Plot Example')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.show()


if __name__ == '__main__':
    main()
