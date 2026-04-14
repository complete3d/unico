import torch

import chamfer
import numpy as np
import torch.nn.functional as F


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferFunction_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2 

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1=None, grad_idx2=None):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2

class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # Use epsilon to avoid infinite gradient at zero distance
        eps = 1e-12
        dist1 = torch.sqrt(dist1 + eps)
        dist2 = torch.sqrt(dist2 + eps)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_PM(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _ = ChamferFunction.apply(xyz1, xyz2)
        eps = 1e-12
        dist1 = torch.sqrt(dist1 + eps)
        return torch.mean(dist1)
    
    
class ChamferDistanceL1_(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, idx1, idx2  = ChamferFunction_.apply(xyz1, xyz2)
        eps = 1e-12
        dist1 = torch.sqrt(dist1 + eps)
        dist2 = torch.sqrt(dist2 + eps)
        return idx1
    

class ChamferDistanceL1_instance2(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, weight):
        """_summary_

        Args:
            xyz1 (list): [16384*3]
            xyz2 (list): [n1*3, n2*3, ..., nN*3]
            weight (list): [16384*1]

        Returns:
            _type_: _description_
        """

        
        lengths2 = [xyz.shape[0] for xyz in xyz2]
        max_length = max(lengths2)
        padded_xyz2 = [torch.cat([xyz, torch.zeros(max_length - len(xyz), 3).to(xyz.device)]) for xyz in xyz2]
        padded_xyz2 = torch.stack(padded_xyz2, dim=0)  # [N, max_length, 3]

        lengths2 = torch.tensor(np.array(lengths2)).to(padded_xyz2.device)  # [N]
        x_mask_2 = (torch.arange(max_length, device=padded_xyz2.device)[None] >= lengths2[:, None])  # shape [N, P1]
        
        xyz1 = [xyz1.clone()[weight[i] > 0.5] for i in range(weight.size(0))]
        lengths1 = [xyz.shape[0] for xyz in xyz1]
        max_length = max(lengths1)
        padded_xyz1 = [torch.cat([xyz, torch.zeros(max_length - len(xyz), 3).to(xyz.device)]) for xyz in xyz1]
        padded_xyz1 = torch.stack(padded_xyz1, dim=0)  
        lengths1 = torch.tensor(np.array(lengths1), device=xyz1[0].device)  # [N]
        x_mask_1 = (torch.arange(max_length, device=xyz1[0].device)[None] >= lengths1[:, None])
        if padded_xyz1.shape[1] == 0:
            # become zero padding
            return torch.ones(lengths1.shape[0], lengths2.shape[0]).to(padded_xyz1.device)
        
        padded_xyz1 = padded_xyz1.unsqueeze(1).expand(-1, padded_xyz2.size(0), -1, -1).reshape(-1, padded_xyz1.size(1), 3)
       
        x_mask_1 = x_mask_1.unsqueeze(1).expand(-1, padded_xyz2.size(0), -1).reshape(-1, padded_xyz1.size(1))
        padded_xyz2 = padded_xyz2.unsqueeze(0).expand(padded_xyz1.size(0), -1, -1, -1).reshape(-1, padded_xyz2.size(1), 3)

        x_mask_2 = x_mask_2.unsqueeze(0).expand(lengths1.shape[0], -1, -1).reshape(-1, padded_xyz2.size(1))
       
        dist1, dist2, idx1, idx2  = ChamferFunction_.apply(padded_xyz1, padded_xyz2) # m, 16384; m, max_length
        eps = 1e-12
        dist1 = torch.sqrt(dist1 + eps)
        dist2 = torch.sqrt(dist2 + eps)
        dist2 = dist2 * (~x_mask_2)
        dist1 = dist1 * (~x_mask_1)
        
        dist = dist1.mean(dim=1) + dist2.mean(dim=1)
        dist = dist.reshape(lengths1.shape[0], lengths2.shape[0])
        return dist

