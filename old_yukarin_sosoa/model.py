from typing import List, Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from old_yukarin_sosoa.config import ModelConfig
from old_yukarin_sosoa.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(
        self,
        f0: List[Tensor],
        phoneme: List[Tensor],
        spec: List[Tensor],
        speaker_id: Optional[List[Tensor]] = None,
    ):
        batch_size = len(spec)

        output1, output2 = self.predictor(
            f0_list=f0,
            phoneme_list=phoneme,
            speaker_id=torch.stack(speaker_id) if speaker_id is not None else None,
        )

        loss1 = F.l1_loss(input=torch.cat(output1), target=torch.cat(spec))
        loss2 = F.l1_loss(input=torch.cat(output2), target=torch.cat(spec))
        loss = loss1 + loss2

        # report
        losses = dict(loss=loss, loss1=loss1, loss2=loss2)
        if not self.training:
            losses = {key: (l, batch_size) for key, l in losses.items()}
        report(losses, self)

        return loss
