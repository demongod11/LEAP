import torch
import torch.nn as nn
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, alpha, class_weights=None):
        super(LossFunction, self).__init__()
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, class_output, reg_output, class_target, reg_target):
        class_raw_loss = nn.BCELoss(reduction='none')(class_output, class_target)
        weighted_loss = class_raw_loss * self.class_weights[:, class_target.long()].squeeze()
        class_loss = weighted_loss.mean()
        new_reg_output = []
        new_reg_target = []
        
        for i in range(len(class_target)):
            if class_target[i].item() == 0:
                new_reg_output.append(reg_output[i].item())
                new_reg_target.append(reg_target[i].item())
        
        if len(new_reg_output) > 0:       
            new_reg_output = torch.tensor(np.array(new_reg_output)).to("cuda:0")
            new_reg_target = torch.tensor(np.array(new_reg_target)).to("cuda:0")
            reg_loss = nn.MSELoss()(new_reg_output, new_reg_target)
        else:
            reg_loss = 0
        
        total_loss = self.alpha * class_loss + (1 - self.alpha) * reg_loss 
        return total_loss
