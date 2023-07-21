import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 

#self.criterion_heatmap = 'l2' nn.MSELoss()
# self.criterion_offset = 'cls' SoftKLLoss()



class SoftKLLoss(nn.KLDivLoss):
    """
        l_n = (Softmax y_n) \cdot \left( \log (Softmax y_n) - \log (Softmax x_n) \right)
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', log_target: bool = False) -> None:
        super(SoftKLLoss,self).__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """  
        input is pred
        target is gt
        """
        input = input.view(-1, input.shape[-2]*input.shape[-1]) # flatten 
        target = target.view(-1, target.shape[-2]*target.shape[-1]) # flatten 
        input = self.logSoftmax(input)
        target = self.Softmax(target)
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
    
    
    
    
def calc_loss(stack_pred_heatmap, gt_heatmap, stack_pred_offset=None,gt_offset=None, 
              criterion=nn.MSELoss(), offset_criterion=SoftKLLoss()):
    """
        calculate the loss for training in one iteration
    """
    # stack_pred_heatmap (5 dim): n_batch,n_stack,n_landmark,heatmap_size,heatmap_size
    # gt_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # calculate the train loss for all stacks
    loss = 0.0 
    stack_gt_heatmap = gt_heatmap.unsqueeze(1).expand([-1,stack_pred_heatmap.size(1)]+list(gt_heatmap.size()[1:])) # [-1, n_stack] + [n_landmark,heatmap_size,heatmap_size]
    loss = criterion(stack_pred_heatmap,stack_gt_heatmap) # MSE between heatmaps
    if offset_criterion is not None:
        loss *= 1 # config.loss_heatmap_weight
        stack_gt_offset = gt_offset.unsqueeze(1).expand([-1,stack_pred_offset.size(1)]+list(gt_offset.size()[1:]))
        loss += 0.05 * offset_criterion(stack_pred_offset,stack_gt_offset) # config.loss_offset_weight
    
    return loss

def calc_inference_loss(pred_heatmap,gt_heatmap,pred_offset=None,gt_offset=None
                        ,criterion=nn.MSELoss(), offset_criterion=SoftKLLoss()):
    """
        calculate the loss for testing in one iteration
    """
    # pred_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # gt_heatmap (4 dim): n_batch,n_landmark,heatmap_size,heatmap_size
    # calculate the loss for the last stack, and the nme
    loss = criterion(pred_heatmap,gt_heatmap)
    if offset_criterion is not None:
        loss *= 1 #config.loss_heatmap_weight
        loss += 0.05  * offset_criterion(pred_offset,gt_offset) # config.loss_offset_weight 
    
    return loss
