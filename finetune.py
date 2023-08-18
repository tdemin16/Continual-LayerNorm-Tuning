from timm import create_model

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from tqdm.auto import tqdm


BATCH_SIZE = 128
DATASET = 'CIFAR100'
EPOCHS = 10
LR = 0.001
MODEL = 'vit_base_patch16_224'
TRAIN_CLS = True


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.05, 1.), ratio=(3./4., 4./3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int((256/224)*224), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    if DATASET == 'CIFAR100':
        train_dataset = datasets.CIFAR100('datasets/', train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100('datasets/', train=False, download=True, transform=val_transform)
    else:
        raise NotImplementedError
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    info = {'n_classes': len(val_dataset.classes)}
    return train_dataloader, val_dataloader, info

def train(model, optim, train_dataloader):
    cum_acc = 0.
    count = 0

    model.train()
    for x, y in tqdm(train_dataloader, leave=False):
        x = x.cuda()
        y = y.cuda()

        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        cum_acc += (torch.argmax(y_hat, dim=-1) == y).float().sum()
        count += x.size(0)

    print(f"Train Acc: {cum_acc / count * 100:.2f}")

@torch.no_grad()
def eval(model, val_dataloader):
    cum_acc = 0.
    count = 0

    model.eval()
    for x, y in tqdm(val_dataloader, leave=False):
        x = x.cuda()
        y = y.cuda()

        y_hat = model(x)
        cum_acc += (torch.argmax(y_hat, dim=-1) == y).float().sum()
        count += x.size(0)

    print(f"Val Acc: {cum_acc / count * 100:.2f}")

def main():
    train_dataloader, val_dataloader, info = get_dataloaders()
    
    model = create_model(
        MODEL,
        pretrained=True,
        num_classes=info['n_classes']
    )
    model = model.cuda()

    params = []
    for n, p in model.named_parameters():
        if 'head' in n or 'norm' in n or ('cls_token' in n and TRAIN_CLS):
            p.requires_grad = True
            params.append((n, p.numel()))
        else:
            p.requires_grad = False

    print("Trainable Params", [n for n, p in params])
    print("Num params", sum([p for n, p in params]))

    lr = LR * BATCH_SIZE / 256.0

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(EPOCHS):
        print(f"Start Epoch: {i}")
        train(model, optim, train_dataloader)
        eval(model, val_dataloader)


if __name__ == '__main__':
    main()