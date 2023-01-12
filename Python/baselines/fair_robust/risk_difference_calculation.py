import numpy as np
import torch

def risk_difference_calculation(Sen_test, pred):
    Sen_test = torch.from_numpy(Sen_test).float()
    # count_S1 = torch.sum(Sen_test)
    # count_S0 = torch.sum(1 - Sen_test)
    count_Y1_S1 = torch.sum(torch.mul(pred, Sen_test))
    count_Y0_S1 = torch.sum(torch.mul(1.0 - pred, Sen_test))
    count_Y1_S0 = torch.sum(torch.mul(pred, 1.0 - Sen_test))
    count_Y0_S0 = torch.sum(torch.mul(1.0 - pred, 1.0 - Sen_test))

    r11 = count_Y1_S1 / len(Sen_test)
    r01 = count_Y0_S1 / len(Sen_test)
    r10 = count_Y1_S0 / len(Sen_test)
    r00 = count_Y0_S0 / len(Sen_test)
    risk_difference = abs(r11 / (r11 + r01) - r10 / (r10 + r00))

    return risk_difference
