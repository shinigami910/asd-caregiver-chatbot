"""
Evaluation metrics used during training.
"""

import math
import torch


def compute_metrics(eval_pred):
    """
    Computes evaluation loss and perplexity.
    """

    logits, labels = eval_pred

    shift_logits = logits[..., :-1, :].reshape(
        -1,
        logits.shape[-1]
    )

    shift_labels = labels[..., 1:].reshape(-1)

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=-100
    )

    loss = loss_function(
        shift_logits,
        shift_labels
    ).item()

    return {

        "eval_loss": loss,

        "perplexity": math.exp(loss)

    }
