import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from logger import setup_logging
from misc.dataset import Load_Dataset
from misc.metrics import _calc_metrics
from misc.utils import get_logger
from models.idea import Idea
from models.exp2 import Exp2
from models.exp3 import Exp3


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
parser.add_argument('--model_name', default='Idea', type=str, help='model name')
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


eeg_train_dataset = torch.load(os.path.join(f'./{args.dataset}/eeg/train.pt'))
eeg_train_dataset = Load_Dataset(eeg_train_dataset)
eeg_train_loader = torch.utils.data.DataLoader(
    dataset=eeg_train_dataset, batch_size=128, shuffle=True, drop_last=False
)

eog_train_dataset = torch.load(os.path.join(f'./{args.dataset}/eog/train.pt'))
eog_train_dataset = Load_Dataset(eog_train_dataset)
eog_train_loader = torch.utils.data.DataLoader(
    dataset=eog_train_dataset, batch_size=128, shuffle=True, drop_last=False
)


device = torch.device(args.device)
model = globals()[f'{args.model_name}']().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-2
)

model.train()
for epoch in range(args.epochs):
    train_loss = []
    train_acc1 = []
    train_acc2 = []
    train_acc3 = []
    for eeg, eog in zip(eeg_train_loader, eog_train_loader):
        eeg, label = eeg[0], eeg[1]
        eog, _ = eog[0], eog[1]
        eeg, eog, label = (
            eeg.float().to(device),
            eog.float().to(device),
            label.long().to(device),
        )
        optimizer.zero_grad()
        pred = model(eeg, eog)
        if len(pred) == 3:
            pred1, pred2, pred3 = pred[0], pred[1], pred[2]
            loss = (
                criterion(pred1, label)
                + criterion(pred2, label)
                + criterion(pred3, label)
            )
            train_acc1.append(
                label.eq(pred1.detach().argmax(dim=1)).float().mean().cpu()
            )
            train_acc2.append(
                label.eq(pred2.detach().argmax(dim=1)).float().mean().cpu()
            )
            train_acc3.append(
                label.eq(pred3.detach().argmax(dim=1)).float().mean().cpu()
            )
        elif len(pred) == 2:
            pred1, pred2 = pred[0], pred[1]
            loss = criterion(pred1, label) + criterion(pred2, label)
            train_acc1.append(
                label.eq(pred1.detach().argmax(dim=1)).float().mean().cpu()
            )
            train_acc2.append(
                label.eq(pred2.detach().argmax(dim=1)).float().mean().cpu()
            )
        else:
            loss = criterion(pred, label)
            train_acc1.append(
                label.eq(pred.detach().argmax(dim=1)).float().mean().cpu()
            )
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    if train_acc3 != []:
        logger.info(
            f"Training->Epoch:{epoch:0>2d}, Loss:{np.mean(train_loss):.3f}, Acc1:{np.mean(train_acc1):.3f}, Acc2:{np.mean(train_acc2):.3f}, Acc3:{np.mean(train_acc3):.3f}"
        )
    elif train_acc2 != []:
        logger.info(
            f"Training->Epoch:{epoch:0>2d}, Loss:{np.mean(train_loss):.3f}, Acc1:{np.mean(train_acc1):.3f}, Acc2:{np.mean(train_acc2):.3f}"
        )
    else:
        logger.info(
            f"Training->Epoch:{epoch:0>2d}, Loss:{np.mean(train_loss):.3f}, Acc:{np.mean(train_acc1):.3f}"
        )


eeg_test_dataset = torch.load(os.path.join(f'./{args.dataset}/eeg/test.pt'))
eeg_test_dataset = Load_Dataset(eeg_test_dataset)
eeg_test_loader = torch.utils.data.DataLoader(
    dataset=eeg_test_dataset, batch_size=128, shuffle=False, drop_last=False
)

eog_test_dataset = torch.load(os.path.join(f'./{args.dataset}/eog/test.pt'))
eog_test_dataset = Load_Dataset(eog_test_dataset)
eog_test_loader = torch.utils.data.DataLoader(
    dataset=eog_test_dataset, batch_size=128, shuffle=False, drop_last=False
)

model.eval()
with torch.no_grad():
    out1 = np.array([])
    out2 = np.array([])
    out3 = np.array([])
    trgs = np.array([])
    for eeg, eog in zip(eeg_test_loader, eog_test_loader):
        eeg, label = eeg[0], eeg[1]
        eog, _ = eog[0], eog[1]
        eeg, eog, label = (
            eeg.float().to(device),
            eog.float().to(device),
            label.long().to(device),
        )
        optimizer.zero_grad()
        pred = model(eeg, eog)

        if len(pred) == 3:
            pred1, pred2, pred3 = pred[0], pred[1], pred[2]

            pred = pred1.max(1, keepdim=True)[1]
            out1 = np.append(out1, pred.cpu().numpy())

            pred = pred2.max(1, keepdim=True)[1]
            out2 = np.append(out2, pred.cpu().numpy())

            pred = pred3.max(1, keepdim=True)[1]
            out3 = np.append(out3, pred.cpu().numpy())

            trgs = np.append(trgs, label.data.cpu().numpy())
        elif len(pred) == 2:
            pred1, pred2 = pred[0], pred[1]

            pred = pred1.max(1, keepdim=True)[1]
            out1 = np.append(out1, pred.cpu().numpy())

            pred = pred2.max(1, keepdim=True)[1]
            out2 = np.append(out2, pred.cpu().numpy())

            trgs = np.append(trgs, label.data.cpu().numpy())
        else:
            pred = pred.max(1, keepdim=True)[1]
            out1 = np.append(out1, pred.cpu().numpy())

            trgs = np.append(trgs, label.data.cpu().numpy())

    if out3 != np.array([]):
        valid_acc_2, valid_f1_2 = _calc_metrics(
            out2, trgs, experiment_log_dir, args.home_path
        )
        valid_acc_3, valid_f1_3 = _calc_metrics(
            out3, trgs, experiment_log_dir, args.home_path
        )
        logger.info(f"Testing_eog->Acc:{valid_acc_2:.3f}, F1:{valid_f1_2:.3f}")
        logger.info(f"Testing_inter->Acc:{valid_acc_3:.3f}, F1:{valid_f1_3:.3f}")

        valid_acc_1, valid_f1_1 = _calc_metrics(
            out1, trgs, experiment_log_dir, args.home_path
        )
        logger.info(f"Testing_eeg->Acc:{valid_acc_1:.3f}, F1:{valid_f1_1:.3f}")
    elif out2 != np.array([]):
        valid_acc_2, valid_f1_2 = _calc_metrics(
            out2, trgs, experiment_log_dir, args.home_path
        )
        logger.info(f"Testing_eog->Acc:{valid_acc_2:.3f}, F1:{valid_f1_2:.3f}")

        valid_acc_1, valid_f1_1 = _calc_metrics(
            out1, trgs, experiment_log_dir, args.home_path
        )
        logger.info(f"Testing_eeg->Acc:{valid_acc_1:.3f}, F1:{valid_f1_1:.3f}")
    else:
        valid_acc_1, valid_f1_1 = _calc_metrics(
            out1, trgs, experiment_log_dir, args.home_path
        )
        logger.info(f"Testing_eeg->Acc:{valid_acc_1:.3f}, F1:{valid_f1_1:.3f}")
