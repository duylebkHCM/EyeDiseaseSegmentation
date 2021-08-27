from typing import Callable, Optional

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from torch import Tensor

__all__ = ["AucPRMetricCallback"]

from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.distributed import all_gather


class AucPRMetricCallback(Callback):
    """
    Auc Precision-Recall score metric
    """

    def __init__(
        self,
        outputs_to_probas: Callable[[Tensor], Tensor] = torch.sigmoid,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc_pr",
        average="macro",
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
        """

        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.outputs_to_probas = outputs_to_probas
        self.y_trues = []
        self.y_preds = []
        self.average = average

    def on_loader_start(self, state):
        self.y_trues = []
        self.y_preds = []

    @torch.no_grad()
    def on_batch_end(self, runner):
        pred_probas = self.outputs_to_probas(runner.output[self.output_key])
        true_labels = runner.input[self.input_key]

        self.y_trues.extend(to_numpy(true_labels))
        self.y_preds.extend(to_numpy(pred_probas))

    def on_loader_end(self, runner):
        y_trues = np.concatenate(all_gather(self.y_trues))
        y_preds = np.concatenate(all_gather(self.y_preds))
        precision, recall, _ = precision_recall_curve(y_trues.reshape(-1), y_preds.reshape(-1))
        score = auc(recall, precision)
        runner.loader_metrics[self.prefix] = float(score)
