# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn


@mmcv.jit(coderize=True)
def accuracy(pred_, target_, domain=False, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    import torch
    import numpy as np
    device = target_.device
    
    if domain==True:
        
        #print(target_)
        index = np.argwhere(  target_.cpu().numpy() < 4 )
        #print(index.shape)
        pred = torch.zeros((index.shape[0],2)).to(device)
        target = torch.zeros((index.shape[0])).to(device)
        pred_ = torch.nn.functional.softmax(pred_, dim=1)
        #print(pred_.shape)
        for i in range(index.shape[0]):
            id = index[i][0]
            pred[i][0] = pred_[id][0]+ pred_[id][1]
            pred[i][1] = pred_[id][2]+ pred_[id][3]
            target[i] = torch.div(target_[i], 2, rounding_mode='floor')
    else:
        pred = pred_
        target = target_
        pred = torch.zeros((pred_.shape[0],3)).to(device)
        pred[:,2] = pred_[:,4]
        pred[:,0] = pred_[:,0]+ pred_[:,2]
        pred[:,1] = pred_[:,1]+ pred_[:,3]
        label2 = (target_)%2
        target = torch.where(target_==4,2,label2)
    
    #print(pred)
    #print(target)
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
