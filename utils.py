import torch
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def train_one_epoch(model, dataloader, optimizer, loss_fun, device):
    model.train()
    for i, (q, diff_level, a, t_used, n_hints, truth, mask) in enumerate(dataloader):
        q = q.to(device)
        diff_level = diff_level.to(device)
        a = a.to(device)
        t_used = t_used.to(device)
        n_hints = n_hints.to(device)
        truth = truth.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        preds = model(q, a, diff_level, n_hints, t_used)
        mask_truth = mask.gt(0).view(-1)
        mask_preds = mask.gt(0.5).view(-1)
        preds = torch.masked_select(preds.view(-1), mask_preds)
        truth = torch.masked_select(truth.view(-1), mask_truth)
        loss = loss_fun(preds, truth)
        loss.backward()
        optimizer.step()

def eval_one_epoch(model, dataloader, device, dataset, epoch):
    model.eval()
    aucs = []
    accs = []
    for i, (q, diff_level, a, t_used, n_hints, truth, mask) in enumerate(dataloader):
        q = q.to(device)
        diff_level = diff_level.to(device)
        a = a.to(device)
        t_used = t_used.to(device)
        n_hints = n_hints.to(device)
        truth = truth.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            preds = model(q, a, diff_level, n_hints, t_used)
        mask_truth = mask.gt(0).view(-1)
        mask_preds = mask.gt(0.5).view(-1)
        preds = torch.masked_select(preds.view(-1), mask_preds).cpu()
        truth = torch.masked_select(truth.view(-1), mask_truth).cpu()

        auc = roc_auc_score(truth, preds)
        acc = accuracy_score(truth, preds.round())

        aucs.append(auc)
        accs.append(acc)

    with open('./checkpoint/%s/performance_record.txt'%dataset, 'a+') as f:
        f.write("epoch%d: auc=%.6f, acc=%.6f\n" % (epoch+1, np.mean(aucs), np.mean(accs)))
    print("auc=%.6f, acc=%.6f"%(np.mean(aucs), np.mean(accs)))