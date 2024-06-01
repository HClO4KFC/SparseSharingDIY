import numpy as np
import os
import omegaconf
import torch.nn
import piq
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from datasets.dataLoader import SingleDataset
from model.mtlModel import build_backbone


class AllTransConvHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AllTransConvHead, self).__init__()
        # 定义转置卷积层
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_transpose4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_transpose5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        # 最终的卷积层用于输出
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 使用转置卷积逐步上采样
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        x = self.conv_transpose3(x)
        x = self.conv_transpose4(x)
        x = self.conv_transpose5(x)
        # 最终卷积层输出预测结果
        x = self.final_conv(x)
        return x


class DepthEstimationModel(torch.nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.backbone, backbone_out_channels = build_backbone('ResNet18')
        self.head = AllTransConvHead(backbone_out_channels, 1)

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
    iter_num = 1
    max_batch = 10
    args = omegaconf.OmegaConf.load(os.path.join('yamls', 'default.yaml'))
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小
        transforms.CenterCrop(224),  # 中心裁剪
        ToFloat16Tensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    output_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToFloat16Tensor()
    ])
    ds_trn = SingleDataset(
        dataset='cityscapes',
        path_pre=os.path.join('..', 'cvDatasets'),
        cv_task_arg=args.cv_tasks_args[0],
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='train',
        in_transform=preprocess,
        out_transform=output_transform,
        label_id_maps={})
    ds_val = SingleDataset(
        dataset='cityscapes',
        path_pre=os.path.join('..', 'cvDatasets'),
        cv_task_arg=args.cv_tasks_args[0],
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='val',
        in_transform=preprocess,
        out_transform=output_transform,
        label_id_maps={})
    trn_loader = DataLoader(
        dataset=ds_trn,
        shuffle=True,
        batch_size=1
    )
    val_loader = DataLoader(
        dataset=ds_val,
        shuffle=False,
        batch_size=10
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthEstimationModel().half().to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.L1Loss()
    trn_loss_sav = []
    val_loss_sav = []
    for iter in range(iter_num):
        print(f'iter{iter}:')
        model.train()

        for batch_idx, (batch_x, batch_std) in enumerate(trn_loader):
            if batch_idx >= max_batch:
                break
            batch_x, batch_std = batch_x.to(device), batch_std.to(device)
            optimizer.zero_grad()
            batch_y = model(batch_x)
            loss = criterion(batch_y, batch_std)
            print(f'batch {batch_idx}, train loss = {loss}')
            trn_loss_sav.append(loss.detach().cpu())
            loss.backward()
            # 梯度裁剪阈值
            clip_value = 1.0
            # 对梯度进行裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            # 查看每层的梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'Layer: {name} | Grad Norm: {param.grad.norm()}')

            optimizer.step()
            scheduler.step()
            # del batch_x, batch_y, batch_std
        model.eval()
        print()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_std) in enumerate(val_loader):
                if batch_idx >= max_batch:
                    break
                batch_y = model(batch_x)
                loss = criterion(batch_y, batch_std)
                print(f'batch {batch_idx}, val loss = {loss}')
                val_loss_sav.append(loss.detach().cpu())

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
