import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from logger import setup_logging
from misc.dataset import Load_Dataset
from misc.metrics import _calc_metrics
from misc.utils import get_logger
from models.exp1 import Exp1


home_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument(
    '--logs_save_dir', default='experiments', type=str, help='saving directory'
)
parser.add_argument(
    '--experiment_description',
    default='Exp',
    type=str,
    help='Experiment Description',
)
parser.add_argument(
    '--run_description', default='run', type=str, help='Running Description'
)
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument(
    '--home_path', default=home_dir, type=str, help='Project home directory'
)
parser.add_argument('--dataset', default='edf20', type=str, help='specify the dataset')
parser.add_argument('--modality', default='eeg', type=str, help='specify the modality')
parser.add_argument(
    '--epochs', default=40, type=int, help='total number of traning epoch'
)
parser.add_argument('--lr', default=3e-4, type=float, help='the inital learning rate')
args = parser.parse_args()


####### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

os.makedirs(args.logs_save_dir, exist_ok=True)
experiment_log_dir = os.path.join(
    args.logs_save_dir,
    args.experiment_description,
    args.run_description,
)
os.makedirs(experiment_log_dir, exist_ok=True)
setup_logging(experiment_log_dir)
logger = get_logger("train")


train_dataset = torch.load(os.path.join(f'./{args.dataset}/{args.modality}/train.pt'))
train_dataset = Load_Dataset(train_dataset)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=128, shuffle=True, drop_last=False
)


device = torch.device(args.device)
model = Exp1().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-2
)

model.train()
for epoch in range(args.epochs):
    train_loss = []
    train_acc = []
    for _, (data, label) in enumerate(train_loader):
        data, label = data.float().to(device), label.long().to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, label)
        train_loss.append(loss.item())
        train_acc.append(label.eq(pred.detach().argmax(dim=1)).float().mean().cpu())
        loss.backward()
        optimizer.step()
    logger.info(
        f"Training->Epoch:{epoch:0>2d}, Loss:{np.mean(train_loss):.3f}, Acc:{np.mean(train_acc):.3f}"
    )


test_dataset = torch.load(os.path.join(f'./{args.dataset}/{args.modality}/test.pt'))
test_dataset = Load_Dataset(test_dataset)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=0
)


model.eval()
with torch.no_grad():
    outs = np.array([])
    trgs = np.array([])
    for data, label in test_loader:
        data, label = data.float().to(device), label.long().to(device)
        pred = model(data)
        pred = pred.max(1, keepdim=True)[1]
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, label.data.cpu().numpy())
    valid_acc, valid_f1 = _calc_metrics(outs, trgs, experiment_log_dir, args.home_path)
    logger.info(f"Testing->Acc:{valid_acc:.3f}, F1:{valid_f1:.3f}")
