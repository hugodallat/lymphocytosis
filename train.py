import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import torch


def train(model, data_loader, criterion, optim, sched=None, architecture='Deep_CNN'):
    columns = ["participant_id", "true_label", "prediction", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    model.train()
    total_loss = 0
    for i, data in enumerate(data_loader):
        optim.zero_grad()

        images, labels = data['images'].cuda(), data['label'].cuda()
        x = torch.cat([data['age'], data['lymph_count']], axis=1).cuda()
        if architecture == "Deep_CNN":
            outputs = model.get_outputs(images[0])
            loss = criterion(outputs, labels)
        elif architecture == "ResNet_with_clinical_attributes":
            outputs = model(images[0], x[0])
            loss = criterion(outputs, labels[0])
        elif architecture == "MOE":
            outputs = model(x[0], images[0])
            loss = criterion(outputs, labels)
        elif architecture == "MLP":
            outputs = model(x[0])
            loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        total_loss += loss.item()
        row = [data['patient'], labels.item(), outputs.item(), 1 if outputs.item() > 0.5 else 0]
        row_df = pd.DataFrame([row], columns=columns)
        results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values, results_df.prediction.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)

    return results_df, results_metrics


def valid(model, data_loader, criterion, architecture = 'Deep_CNN'):
    model.eval()
    columns = ["participant_id", "true_label", "prediction", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data['images'].cuda(), data['label'].cuda()
            x = torch.cat([data['age'], data['lymph_count']], axis=1).cuda()
            if architecture == "Deep_CNN":
                outputs = model.get_outputs(images[0])
                loss = criterion(outputs, labels)
            elif architecture == "ResNet_with_clinical_attributes":
                outputs = model(images[0], x[0])
                loss = criterion(outputs, labels[0])
            elif architecture == "MOE":
                outputs = model(x[0], images[0])
                loss = criterion(outputs, labels)
            elif architecture == "MLP":
                outputs = model(x[0])
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            row = [data['patient'], labels.item(), outputs.item(), 1 if outputs.item() > 0.5 else 0]
            row_df = pd.DataFrame([row], columns=columns)
            results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values, results_df.prediction.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)

    return results_df, results_metrics


def compute_metrics(ground_truth, predicted_label, prediction):
    """Computes the accuracy, sensitivity, specificity and balanced accuracy"""
    tp = np.sum((predicted_label == 1) & (ground_truth == 1))
    tn = np.sum((predicted_label == 0) & (ground_truth == 0))
    fp = np.sum((predicted_label == 1) & (ground_truth == 0))
    fn = np.sum((predicted_label == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity/ Recall
    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    else:
        metrics_dict['sensitivity'] = 0.0

    # Specificity
    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)
    else:
        metrics_dict['specificity'] = 0.0

    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2
    fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    roc_auc = auc(fpr, tpr)
    metrics_dict['AUC'] = roc_auc
    return metrics_dict


def lr_decay(step, decay_rate=0.1):
    if step < 3000:
        lambda_lr = 1
    elif step < 10000:
        lambda_lr = decay_rate
    elif step < 20000:
        lambda_lr = decay_rate * decay_rate
    return lambda_lr


def eval(model, data_loader, architecture):
    model.eval()
    columns = ["ID" ,"Predicted"]
    results_df = pd.DataFrame(columns=columns)
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images = data['images'].cuda()
            x = torch.cat([data['age'], data['lymph_count']] ,axis=1).cuda()
            if architecture == "Deep_CNN":
                outputs = model.get_outputs(images[0])
            elif architecture == "ResNet_with_clinical_attributes":
                outputs = model(images[0], x[0])
            elif architecture == "MOE":
                outputs = model(x[0], images[0])
            elif architecture == "MLP":
                outputs = model(x[0])
            row = [data['patient'][0], 1 if outputs.item() > 0.5 else 0]
            row_df = pd.DataFrame([row], columns=columns)
            results_df = pd.concat([results_df, row_df])

    return results_df