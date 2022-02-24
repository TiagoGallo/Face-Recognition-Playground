import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from prettytable import PrettyTable

def eval_model(test_loader, face_rec_model):
    table = PrettyTable(['Dataset', 'ACC', 'AUC', 'mAP'])
    face_rec_model.eval()

    accuracies = []

    if isinstance(test_loader, list):
        for loader in test_loader:
            auc, mAP, acc = evaluation(loader, face_rec_model, loader.dataset.name)
            accuracies.append(acc)
            table.add_row([loader.dataset.name, f'{acc*100:.2f}%', f'{auc*100:.2f}%', f'{mAP*100:.2f}%'])
    else:
        auc, mAP, acc = evaluation(test_loader, face_rec_model, test_loader.dataset.name)
        accuracies.append(acc)
        table.add_row([test_loader.dataset.name, f'{acc*100:.2f}%', f'{auc*100:.2f}%', f'{mAP*100:.2f}%'])

    print(table)

    return np.average(accuracies)

def evaluation(test_loader, face_rec_model, dataset_name=None):
    features = []

    with torch.no_grad():
        for images in tqdm(test_loader,
                            total=len(test_loader),
                            desc=f'Evaluating Dataset {dataset_name if dataset_name is not None else ""}'):
            images = images.cuda()
            feat = face_rec_model(images)
            feat = feat.data.cpu()
            features.append(feat)

    features_stack = torch.cat(features, dim=0)
    features = F.normalize(features_stack, p=2, dim=1)  # L2-normalize

    num_feat = features.size()[0]
    feat_pair1 = features[np.arange(0, num_feat, 2), :]
    feat_pair2 = features[np.arange(1, num_feat, 2), :]
    feat_dist = F.cosine_similarity(feat_pair1, feat_pair2).numpy()

    # Eval metrics
    gt = np.asarray(test_loader.dataset.issame_list)

    roc_auc = roc_auc_score(gt, feat_dist)
    mAP = average_precision_score(gt, feat_dist)
    best_acc, best_thres = calculate_accuracy(feat_dist, gt)

    return roc_auc, mAP, best_acc

def calculate_accuracy(dist, actual_issame):
    best_acc = 0
    best_threshold = 0
    for threshold in np.arange(0, 1, 0.001):
        predict_issame = np.greater(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        # fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(
            np.logical_and(np.logical_not(predict_issame),
                        np.logical_not(actual_issame)))
        # fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        acc = float(tp + tn) / dist.size

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_acc, best_threshold
