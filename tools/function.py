import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import pynvml
import torch.nn.functional as F
from tqdm import tqdm
import random

np.seterr(divide='ignore', invalid='ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():

        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        gpu_ids.append(dst_gpu_id)

    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()

    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device


def multiclass_acc(y_pred, y_true):
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

def eval_fgmsa_regression(y_pred, y_true):
    test_preds = y_pred.view(-1).cpu().detach().numpy()
    test_truth = y_true.view(-1).cpu().detach().numpy()
    test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

    ms_2 = [-1.01, 0.0, 1.01]
    test_preds_a2 = test_preds.copy()
    test_truth_a2 = test_truth.copy()
    for i in range(2):
        test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
    for i in range(2):
        test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

    ms_3 = [-1.01, -0.3, 0.3, 1.01]
    test_preds_a3 = test_preds.copy()
    test_truth_a3 = test_truth.copy()
    for i in range(3):
        test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
    for i in range(3):
        test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

    ms_5 = [-1.01, -0.7, -0.3, 0.3, 0.7, 1.01]
    test_preds_a5 = test_preds.copy()
    test_truth_a5 = test_truth.copy()
    for i in range(5):
        test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
    for i in range(5):
        test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)

    mae = np.mean(np.absolute(test_preds - test_truth)).astype(np.float64)
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a2 = multiclass_acc(test_preds_a2, test_truth_a2)
    mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

    eval_results = {
        "Mult_acc_2": round(mult_a2, 4),
        "Mult_acc_3": round(mult_a3, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "Mult_acc_7": round(mult_a7, 4),
        "F1_score": round(f_score, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4),
    }
    return eval_results

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query, positive_key, negative_keys):
        """
        Args:
        - query: Tensor of shape (batch_size, dim) - the query embeddings
        - positive_key: Tensor of shape (batch_size, dim) - the positive key embeddings
        - negative_keys: Tensor of shape (batch_size, num_negatives, dim) - the negative key embeddings

        Returns:
        - loss: Tensor - the computed InfoNCE loss
        """
        # Normalize the query and key embeddings
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        negative_keys = F.normalize(negative_keys, dim=-1)

        # Compute positive logits
        positive_logits = torch.sum(query * positive_key, dim=-1, keepdim=True) / self.temperature

        # Compute negative logits
        negative_logits = torch.bmm(query.unsqueeze(1), negative_keys.permute(0, 2, 1)).squeeze(1) / self.temperature

        # Concatenate positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # Create labels for the positive samples (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Compute the loss using cross-entropy
        loss = F.cross_entropy(logits, labels)

        return loss