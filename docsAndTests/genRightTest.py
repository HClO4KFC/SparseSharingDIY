import torch
import torch.nn as nn
import torchvision.models as models

from omegaconf import omegaconf
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.dataLoader import SingleDataset


class EyeImageGenerator(nn.Module):
    def __init__(self):
        super(EyeImageGenerator, self).__init__()
        # 使用预训练的ResNet50作为特征提取网络
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除ResNet50的分类层
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # 解码器：从特征图生成图像
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Tanh()  # 输出图像像素值在[-1, 1]之间
        )

    def forward(self, x):
        # 提取特征
        x = self.encoder(x)
        # 生成图像
        x = self.decoder(x)
        return x


# 测试模型
if __name__ == "__main__":
    # 创建模型并移动到GPU
    model = EyeImageGenerator().cuda()
    args = omegaconf.OmegaConf.load('../yamls/default.yaml')
    cv_task_arg = [cv_task_arg for cv_task_arg in args.cv_tasks_args if cv_task_arg.name == 'genRightCam']
    train_set = SingleDataset(
        dataset='cityspaces',
        path_pre='../../cvDatasets',
        cv_task_arg=cv_task_arg,
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='train',
        transform=transforms.Compose([transforms.ToTensor()]),
        label_id_maps={args.cv_tasks_args[i].output:args.cv_tasks_args[i].label_id_map for i in range(len(args.cv_tasks_args))})
    val_set = SingleDataset(
        dataset='cityscapes',
        path_pre='..\cvDatasets',
        cv_task_arg=cv_task_arg,
        cv_subsets_args=args.cv_subsets_args,
        train_val_test='val',
        transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=8,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=8,
        shuffle=False
    )
    # input_tensor = torch.randn(32, 3, 224, 224).cuda()  # 批量大小为32的输入图像
    # target_tensor = torch.randn(32, 3, 224, 224).cuda()  # 假设目标图像大小相同

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = 'cuda:0'

    epoch_num = 20
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{epoch_num}], Batch [{batch_idx+1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)