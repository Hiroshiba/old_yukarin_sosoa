from typing import Optional, Sequence

import numpy
import torch
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from old_yukarin_sosoa.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        hidden_size: int,
        block_num: int,
    ):
        super().__init__()
        self.output_size = output_size

        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        input_size = input_feature_size + speaker_embedding_size
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = Encoder(
            idim=None,
            attention_dim=hidden_size,
            attention_heads=2,
            linear_units=hidden_size * 4,
            num_blocks=block_num,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            normalize_before=True,
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        self.post = torch.nn.Linear(hidden_size, output_size)

        self.postnet = Postnet(
            idim=None,
            odim=output_size,
            n_layers=5,
            n_chans=hidden_size,
            n_filts=5,
            use_batch_norm=True,
            dropout_rate=0.5,
        )

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        f0_list: Sequence[Tensor],
        phoneme_list: Sequence[Tensor],
        speaker_id: Optional[Tensor],
    ):
        length_list = [f0.shape[0] for f0 in f0_list]

        length = torch.from_numpy(numpy.array(length_list)).to(f0_list[0].device)
        f0 = pad_sequence(f0_list, batch_first=True)
        phoneme = pad_sequence(phoneme_list, batch_first=True)

        h = torch.cat((f0, phoneme), dim=2)  # (batch_size, length, ?)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            h = torch.cat((h, speaker_feature), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return (
            [output1[i, :l] for i, l in enumerate(length_list)],
            [output2[i, :l] for i, l in enumerate(length_list)],
        )

    def inference(
        self,
        f0_list: Sequence[Tensor],
        phoneme_list: Sequence[Tensor],
        speaker_id: Optional[Tensor],
    ):
        _, h = self(f0_list=f0_list, phoneme_list=phoneme_list, speaker_id=speaker_id)
        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        input_feature_size=config.input_feature_size,
        output_size=config.output_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        hidden_size=config.hidden_size,
        block_num=config.block_num,
    )
