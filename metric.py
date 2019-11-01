import torch

def clf_acc(yhat, y, pad_id=0):
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float()[y != pad_id].mean() # padding은 acc에서 제거
    return acc

def sim_acc(yhat, y):
    with torch.no_grad():
        acc = (yhat == y).float().mean() # padding은 acc에서 제거
    return acc

def correct_sum(y_pred, dec_output):
    with torch.no_grad():
        y_pred = y_pred.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
        correct_elms = (y_pred == dec_output).float()[dec_output != 0]
        correct_sum = correct_elms.sum().to(torch.device('cpu')).numpy()  # padding은 acc에서 제거
        num_correct_elms = len(correct_elms)
    return correct_sum, num_correct_elms


if __name__ == '__main__':
    pass
