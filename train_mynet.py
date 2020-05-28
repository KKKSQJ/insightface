from model import MyfaceNet
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
import argparse
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import tqdm


def get_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_loader(args):
    ds, class_num = get_dataset(args.train_datapath)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=3)
    return loader, class_num

def get_val_loader(args):
    ds, class_num = get_dataset(args.val_datapath)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=3)
    return loader, class_num

def train(args):
    train_loader, train_class_num = get_train_loader(args)
    val_loader, val_class_num = get_val_loader(args)


    model = MyfaceNet(train_class_num) #train_class_num

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        #for batch_x, batch_y in tqdm(iter(loader)):
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, args.epochs, batch, train_class_num / args.batch_size,
                     loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / train_class_num / args.batch_size,
                                               train_acc / train_class_num))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            if args.cuda:
                batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())
            else:
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / val_class_num / args.batch_size,
                                             eval_acc / val_class_num))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'work_space/models/params_' + str(epoch + 1) + '.pth')
            # to_onnx(model, 3, 28, 28, 'params.onnx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--train_datapath', required=True, help='train data path')
    parser.add_argument('--val_datapath', required=True, help='val data path')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--use_cuda', default=False, help='using CUDA for training')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    train(args)