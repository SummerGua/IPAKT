import torch
from sklearn.metrics import roc_auc_score, accuracy_score

def train_one_epoch(model, dataloader, optimizer, loss_fun, device):
    model.train()
    for i, (q, diff_level, a, t_used, n_hints, gap, truth, mask) in enumerate(dataloader):
        q = q.to(device)
        diff_level = diff_level.to(device)
        a = a.to(device)
        t_used = t_used.to(device)
        n_hints = n_hints.to(device)
        gap = gap.to(device)
        truth = truth.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        preds = model(q, a, diff_level, n_hints, gap, t_used)
        mask_truth = mask.gt(0).view(-1)
        mask_preds = mask.gt(0.5).view(-1)
        preds = torch.masked_select(preds.view(-1), mask_preds)
        truth = torch.masked_select(truth.view(-1), mask_truth)
        loss = loss_fun(preds, truth)
        loss.backward()
        optimizer.step()

def eval_one_epoch(model, dataloader, device):
    model.eval()
    for i, (q, diff_level, a, t_used, n_hints, gap, truth, mask) in enumerate(dataloader):
        q = q.to(device)
        diff_level = diff_level.to(device)
        a = a.to(device)
        t_used = t_used.to(device)
        n_hints = n_hints.to(device)
        gap = gap.to(device)
        truth = truth.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            preds = model(q, a, diff_level, n_hints, gap, t_used)
        mask_truth = mask.gt(0).view(-1)
        mask_preds = mask.gt(0.5).view(-1)
        preds = torch.masked_select(preds.view(-1), mask_preds)
        truth = torch.masked_select(truth.view(-1), mask_truth)

    auc = roc_auc_score(truth, preds)
    acc = accuracy_score(truth, preds.round())

    print("auc=%.6f, acc=%.6f"%(auc, acc))