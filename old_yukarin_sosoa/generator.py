from pathlib import Path
from typing import Any, List, Union

import numpy
import torch
from torch import Tensor

from old_yukarin_sosoa.config import Config
from old_yukarin_sosoa.network.predictor import Predictor, create_predictor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        f0_list: List[Union[numpy.ndarray, torch.Tensor]],
        phoneme_list: List[Union[numpy.ndarray, torch.Tensor]],
        speaker_id: Union[numpy.ndarray, torch.Tensor] = None,
    ):
        f0_list = [to_tensor(f0).to(self.device) for f0 in f0_list]
        phoneme_list = [to_tensor(phoneme).to(self.device) for phoneme in phoneme_list]
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id).to(self.device)

        with torch.no_grad():
            output_list = self.predictor.inference(
                f0_list=f0_list, phoneme_list=phoneme_list, speaker_id=speaker_id
            )
        return [output.cpu().numpy() for output in output_list]
