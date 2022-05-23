import torch
import tqdm

import constants

def get_preds(dataloader, model):
    """ Eval on data

    Args:
        dataloader (torch.Dataloader)
        model (torch.nn.Module)

    Returns:
        (torch.Tensor) [nb_samples] : True labels
        (torch.Tensor) [nb_samples, nb_classes] : logits

    """

    model.eval()
    y_true = []
    logits = []

    # Iterate over the dataset
    for (X, y) in tqdm.tqdm(dataloader):

        with torch.no_grad():

            # Work with the GPU if available
            X, y = X.to(constants.device), y.to(constants.device)

            # Compute prediction error
            preds = model(X)
            y_true.append(y)
            logits.append(preds)
    y_true = torch.cat(y_true)
    logits = torch.cat(logits)

    return y_true, logits

