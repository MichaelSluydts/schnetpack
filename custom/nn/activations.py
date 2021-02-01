import numpy as np
from torch.nn import functional as F


def NegLogSigmoid(x):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """
    return -F.logsigmoid(-x)
    
def ShiftedELU(x):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """
    return F.elu(x)+1.0