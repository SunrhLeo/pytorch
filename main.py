import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import sys
import copy
import json
from PIL import Image
import pylab


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    print(torch.__version__)

""":type tensor torch.Tensor"""


def im_convert(tensor: torch.Tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def set_parameter_require_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


# 初始化模型
def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_require_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.LogSoftmax(dim=1))
        input_size = 224
    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_require_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size


# 模型训练
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename="defaultTrainName.pth"):
    since = time.time()
    best_acc = 0.0
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()

                # 正向
                # 只有训练的时候才更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_output = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_output, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('学习率：{:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['test']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return model.train(mode=was_training)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = './flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_trasforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),  # 随机旋转45度，-45到45之间随机旋转
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.RandomHorizontalFlip(0.5),  # 随机水平翻转，参数为概率
            transforms.RandomVerticalFlip(0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，2为对比度 3为饱和度 4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转为灰度图，依然保留3通道，R=G=B
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])
    }

    batch_size = 8

    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_trasforms[x]) for x in ['train', 'valid']}
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    # print(class_names)
    # print(image_datasets)
    # 读取标签，编号与花名的映射关系
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # 检测数据是否正确
    # images, labels = next(iter(data_loaders['train']))
    # print(len(images[0, 2]))
    # plt.imshow(images[0, 0])
    # pylab.show()

    # 查看数据
    # fig = plt.figure(figsize=(20, 12))
    # columns = 4
    # rows = 2
    #
    # data_iter = iter(data_loaders['valid'])
    # inputs, classes = next(data_iter)
    #
    # for idx in range(columns * rows):
    #     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    #     ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    #     plt.imshow(im_convert(inputs[idx]))
    #
    # plt.show()
    # pylab.show()

    model_name = 'resnet'
    # 是否用人家训练好的特征来做
    feature_extract = True

    # 是否用GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print("CUDA is not available, on CPU now...")
    else:
        print("CUDA is available, on GPU now...")

    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    # model_ft = models.resnet152()
    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
    model_ft = model_ft.to(device)

    # 模型保存
    filename = 'checkpoint.pth'

    # 是否训练所有层
    params_to_update = model_ft.parameters()
    print("Param To Learn:")
    if feature_extract:
        # params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # 优化器
    optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
    # 学习率没7个epochs衰减为原来的1/10
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # 损失函数：最后一层为LogSoftmax，所以不能用CrossEntropyLoss计算，CrossEntropyLoss相当于softMax与NLLLoss整合
    criterion = nn.NLLLoss()

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, data_loaders, criterion, optimizer_ft, num_epochs=20, is_inception=False, filename=filename)

