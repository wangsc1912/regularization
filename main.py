import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import utils
from models import MLP, CNN
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'])
parser.add_argument('--positive_regularization', type=float, default=0.0, help='regularization. 0.0 means no regularization')
args = parser.parse_args()

# datasets
transform = transforms.Compose([transforms.ToTensor(),])
tr_dataset = MNIST(root='data', train=True, transform=transform, download=True)
te_dataset = MNIST(root='data', train=False, transform=transform, download=True)

tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
te_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
if args.model == 'cnn':
    model = CNN().to(device)
elif args.model == 'mlp':
    model = MLP(input_size=784, hidden_size=100, output_size=10).to(device)

# regularization factor
reg_factor = args.positive_regularization

# loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler

# train
tr_acc, te_acc = [], []
for epoch in range(1, 101):
    for i, (x, y) in enumerate(tr_loader):
        batchsize_current = x.shape[0]
        x = x.to(device)
        if args.model == 'mlp':
            x = x.view(batchsize_current, -1)
        y = y.to(device)
        output = model(x)

        acc_batch = (output.argmax(dim=1) == y.to(device)).float().mean().item()
        tr_acc.append(acc_batch)
        pos_reg = utils.pos_regularization(model)
        loss = criterion(output, y) + reg_factor * pos_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for i, (x, y) in enumerate(te_loader):
            batchsize_current = x.shape[0]
            x = x.to(device)
            if args.model == 'mlp':
                x = x.view(batchsize_current, -1)
            y = y.to(device)
            output = model(x)
            acc_batch = (output.argmax(dim=1) == y.to(device)).float().mean().item()
            te_acc.append(acc_batch)

    tr_acc_epoch = sum(tr_acc) / len(tr_acc)
    te_acc_epoch = sum(te_acc) / len(te_acc)
    print(f'epoch: {epoch}, tr_acc: {tr_acc_epoch}, te_acc: {te_acc_epoch}')
