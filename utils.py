import torch

def pos_regularization(model):
    pos_reg = None
    for name, weight in model.named_parameters():
        # if 'weight' in name and 'conv' in name: 
        if 'weight' in name: 
            # reg = (torch.exp(-10 * weight) - 1).sum()
            # torch.where(weight>0, 0, weight)
            reg = torch.relu(-weight)
            reg = reg.sum()
            # if reg < 0:
                # reg = 0
            if pos_reg is None:
                pos_reg = reg
            else:
                pos_reg += reg
    return pos_reg
