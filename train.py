import os
import warnings
from model.FPN import FPN
import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
from utils.dataset import loader
torch.manual_seed(42)
warnings.filterwarnings("ignore")

bs = 32

train_loader, val_loader = loader(batch_size=bs)

model = FPN([2, 4, 23, 3], 19).cuda()

# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)
# 选用adam优化器来训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5, last_epoch=-1)

# 训练50轮
epochs_num = 50


def trainer(net, train_loader, val_loader, loss, trainer, num_epochs, scheduler,
            devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_loader)

    # 这一行导致在训练中画图，暂时不改
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        print("starting epoch: ", epoch + 1)
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_loader):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, val_loader)
        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()
        print(
            f"epoch {epoch+1} --- loss {metric[0] / metric[2]:.3f} ---  train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch+1)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        # 保存训练数据 make sure the path exists
        if not os.path.exists('data'):
            os.makedirs('data')
        df.to_excel("data/FPN.xlsx")
        # ----------------保存模型-------------------
        # makedir checkpoint
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        if np.mod(epoch+1, 5) == 0:
            torch.save(model.state_dict(),
                       "checkpoints/FPN_{}.pth".format(epoch+1))


if __name__ == "__main__":
    trainer(model, train_loader, val_loader, lossf, optimizer, epochs_num,
            scheduler)
