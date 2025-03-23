from typing import List, Optional

import numpy
import torch
from pytorch_trainer import report
from torch import Tensor, nn

from old_yukarin_sosoa.generator import Generator


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
    ):
        super().__init__()
        self.generator = generator

    def __call__(
        self,
        f0: List[Tensor],
        phoneme: List[Tensor],
        spec: List[Tensor],
        speaker_id: Optional[List[Tensor]] = None,
    ):
        batch_size = len(spec)

        output = self.generator.generate(
            f0_list=f0,
            phoneme_list=phoneme,
            speaker_id=torch.stack(speaker_id) if speaker_id is not None else None,
        )

        diff = numpy.abs(
            numpy.concatenate(output)
            - numpy.concatenate([s.cpu().numpy() for s in spec])
        ).mean()

        scores = {"diff": (diff, batch_size)}

        report(scores, self)
        return scores
