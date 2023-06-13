import torch

def confidence_penalty(preds_softmax, apply_non_lin):

    eps = 1e-5
    if apply_non_lin is not None:
        preds_softmax = apply_non_lin(preds_softmax)
    # dims = len(preds_softmax.shape)-2
    loss = preds_softmax * torch.clamp(torch.log(preds_softmax), min=eps, max=1-eps) 
    # Hopefully it's never log(1) -> Murphy's law

    loss = torch.clamp(torch.sum(preds_softmax, dim=0), min=-1e-5, max=1e-5)
    loss = -1.0 * loss.mean(0)

    return loss