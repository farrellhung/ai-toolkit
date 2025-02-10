import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

def orthogonal_regularization(input, lambda_ortho=8e-8):
    if lambda_ortho <= 0:
        return 0
    
    transpose = torch.transpose(input,2,3)
    inner = torch.matmul(transpose, input)
    eye = torch.eye(inner.size(2), device=input.device)
    
    ortho_loss = torch.norm(inner - eye, p='fro')
    ortho_loss = (ortho_loss**2)*lambda_ortho

    f = open('ortho_loss.txt', 'a')
    f.write("{:.6f}".format(float(ortho_loss))+'\n')
    f.close()

    return ortho_loss

def basis_regularization(input, lambda_basis=4e-7):
    if lambda_basis <= 0:
        return 0
    
    basis_loss = torch.sum((torch.sum(input**2, dim=2) - 1)**2)
    basis_loss = basis_loss*lambda_basis

    f = open('basis_loss.txt', 'a')
    f.write("{:.6f}".format(float(basis_loss))+'\n')
    f.close()

    return basis_loss