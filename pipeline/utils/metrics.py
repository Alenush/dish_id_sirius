import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def update_error_types(error_types, y_pred, y_true):
  error_types['tp_i'] += (y_pred * y_true).sum(0).cpu().data.numpy()
  error_types['fp_i'] += (y_pred * (1-y_true)).sum(0).cpu().data.numpy()
  error_types['fn_i'] += ((1-y_pred) * y_true).sum(0).cpu().data.numpy()
  error_types['tn_i'] += ((1-y_pred) * (1-y_true)).sum(0).cpu().data.numpy()

  error_types['tp_all'] += (y_pred * y_true).sum().item()
  error_types['fp_all'] += (y_pred * (1-y_true)).sum().item()
  error_types['fn_all'] += ((1-y_pred) * y_true).sum().item()

def label2onehot(labels, pad_value=len(word2id)-1):
    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad and eos position
    one_hot = one_hot[:, 1:-1]
    one_hot[:, 0] = 0

    return one_hot

def compute_metrics(ret_metrics, error_types, metric_names, eps=1e-10, weights=None):
    if 'accuracy' in metric_names:
        ret_metrics['accuracy'].append(np.mean((error_types['tp_i'] + error_types['tn_i']) / (error_types['tp_i'] + error_types['fp_i'] + error_types['fn_i'] + error_types['tn_i'])))
    if 'jaccard' in metric_names:
        ret_metrics['jaccard'].append(error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all'] + eps))
    if 'dice' in metric_names:
        ret_metrics['dice'].append(2*error_types['tp_all'] / (2*(error_types['tp_all'] + error_types['fp_all'] + error_types['fn_all']) + eps))
    if 'f1' in metric_names:
        pre = error_types['tp_i'] / (error_types['tp_i'] + error_types['fp_i'] + eps)
        rec = error_types['tp_i'] / (error_types['tp_i'] + error_types['fn_i'] + eps)
        f1_perclass = 2*(pre * rec) / (pre + rec + eps)
        if 'f1_ingredients' not in ret_metrics.keys():
            ret_metrics['f1_ingredients'] = [np.average(f1_perclass, weights=weights)]
        else:
            ret_metrics['f1_ingredients'].append(np.average(f1_perclass, weights=weights))

        pre = error_types['tp_all'] / (error_types['tp_all'] + error_types['fp_all'] + eps)
        rec = error_types['tp_all'] / (error_types['tp_all'] + error_types['fn_all'] + eps)
        f1 = 2*(pre * rec) / (pre + rec + eps)
        ret_metrics['f1'].append(f1)