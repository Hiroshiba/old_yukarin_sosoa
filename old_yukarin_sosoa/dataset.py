import json
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from old_yukarin_sosoa.config import DatasetConfig
from old_yukarin_sosoa.data.phoneme import OjtPhoneme
from old_yukarin_sosoa.data.sampling_data import SamplingData

mora_phoneme_list = ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl", "pau"]
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by", "gw"]
)


class F0ProcessMode(str, Enum):
    normal = "normal"
    phoneme_mean = "phoneme_mean"
    mora_mean = "mora_mean"
    voiced_mora_mean = "voiced_mora_mean"


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: Optional[numpy.ndarray],
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    if weight is None:
        for a in numpy.split(f0, indexes):
            a[:] = numpy.mean(a[a > 0])
    else:
        for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes)):
            a[:] = numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
    f0[numpy.isnan(f0)] = 0
    return f0


def get_notsilence_range(silence: numpy.ndarray, prepost_silence_length: int):
    """
    最初と最後の無音を除去したrangeを返す。
    一番最初や最後が無音でない場合はノイズとみなしてその区間も除去する。
    最小でもprepost_silence_lengthだけは確保する。
    """
    length = len(silence)

    ps = numpy.argwhere(numpy.logical_and(silence[:-1], ~silence[1:]))
    pre_length = ps[0][0] + 1 if len(ps) > 0 else 0
    pre_index = max(0, pre_length - prepost_silence_length)

    ps = numpy.argwhere(numpy.logical_and(~silence[:-1], silence[1:]))
    post_length = length - (ps[-1][0] + 1) if len(ps) > 0 else 0
    post_index = length - max(0, post_length - prepost_silence_length)
    return range(pre_index, post_index)


@dataclass
class Input:
    f0: SamplingData
    phoneme: SamplingData
    spec: SamplingData
    silence: SamplingData
    phoneme_list: Optional[List[OjtPhoneme]]
    volume: Optional[SamplingData]


@dataclass
class LazyInput:
    f0_path: Path
    phoneme_path: Path
    spec_path: Path
    silence_path: Path
    phoneme_list_path: Optional[Path]
    volume_path: Optional[Path]

    def generate(self):
        return Input(
            f0=SamplingData.load(self.f0_path),
            phoneme=SamplingData.load(self.phoneme_path),
            spec=SamplingData.load(self.spec_path),
            silence=SamplingData.load(self.silence_path),
            phoneme_list=(
                self.phoneme_class.load_julius_list(
                    self.phoneme_list_path, verify=False
                )
                if self.phoneme_list_path is not None
                else None
            ),
            volume=(
                SamplingData.load(self.volume_path)
                if self.volume_path is not None
                else None
            ),
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: Sequence[Union[Input, LazyInput]],
        prepost_silence_length: int,
        max_sampling_length: int,
        f0_process_mode: F0ProcessMode,
        time_mask_max_second: float,
        time_mask_rate: float,
    ):
        self.inputs = inputs
        self.prepost_silence_length = prepost_silence_length
        self.max_sampling_length = max_sampling_length
        self.f0_process_mode = f0_process_mode
        self.time_mask_max_second = time_mask_max_second
        self.time_mask_rate = time_mask_rate

    @staticmethod
    def extract_input(
        f0_data: SamplingData,
        phoneme_data: SamplingData,
        spec_data: SamplingData,
        silence_data: SamplingData,
        phoneme_list_data: Optional[List[OjtPhoneme]],
        volume_data: Optional[SamplingData],
        prepost_silence_length: int,
        max_sampling_length: int,
        f0_process_mode: F0ProcessMode,
        time_mask_max_second: float,
        time_mask_rate: float,
    ):
        rate = spec_data.rate

        f0 = f0_data.resample(rate)
        phoneme = phoneme_data.resample(rate)
        silence = silence_data.resample(rate)
        volume = volume_data.resample(rate) if volume_data is not None else None
        spec = spec_data.array

        assert numpy.abs(len(spec) - len(f0)) < 5
        assert numpy.abs(len(spec) - len(phoneme)) < 5
        assert numpy.abs(len(spec) - len(silence)) < 5
        assert volume is None or numpy.abs(len(spec) - len(silence)) < 5

        length = min(len(spec), len(f0), len(phoneme), len(silence))
        if volume is not None:
            length = min(length, len(volume))

        # 最初と最後の無音を除去する
        notsilence_range = get_notsilence_range(
            silence=silence[:length],
            prepost_silence_length=prepost_silence_length,
        )
        f0 = f0[notsilence_range]
        silence = silence[notsilence_range]
        phoneme = phoneme[notsilence_range]
        spec = spec[notsilence_range]
        if volume is not None:
            volume = volume[notsilence_range]
        length = len(f0)

        # サンプリング長調整
        if length > max_sampling_length:
            offset = numpy.random.randint(length - max_sampling_length + 1)
            offset_slice = slice(offset, offset + max_sampling_length)
            f0 = f0[offset_slice]
            silence = silence[offset_slice]
            phoneme = phoneme[offset_slice]
            spec = spec[offset_slice]
            if volume is not None:
                volume = volume[offset_slice]
            length = max_sampling_length

        if f0_process_mode == F0ProcessMode.normal:
            pass
        else:
            assert phoneme_list_data is not None
            weight = volume

            if f0_process_mode == F0ProcessMode.phoneme_mean:
                split_second_list = [p.end for p in phoneme_list_data[:-1]]
            else:
                split_second_list = [
                    p.end
                    for p in phoneme_list_data[:-1]
                    if p.phoneme in mora_phoneme_list
                ]

            if f0_process_mode == F0ProcessMode.voiced_mora_mean:
                if weight is None:
                    weight = numpy.ones_like(f0)

                for p in phoneme_list_data:
                    if p.phoneme not in voiced_phoneme_list:
                        weight[int(p.start * rate) : int(p.end * rate)] = 0

            weight = weight[:length]

            f0 = f0_mean(
                f0=f0,
                rate=rate,
                split_second_list=split_second_list,
                weight=weight,
            )

        if silence.ndim == 2:
            silence = numpy.squeeze(silence, axis=1)

        if time_mask_max_second > 0 and time_mask_rate > 0:
            expected_num = time_mask_rate * length
            num = int(expected_num) + int(
                numpy.random.rand() < (expected_num - int(expected_num))
            )
            for _ in range(num):
                mask_length = numpy.random.randint(int(rate * time_mask_max_second))
                mask_offset = numpy.random.randint(len(f0) - mask_length + 1)
                f0[mask_offset : mask_offset + mask_length] = 0
                phoneme[mask_offset : mask_offset + mask_length] = 0

        return dict(
            f0=f0.astype(numpy.float32),
            phoneme=phoneme.astype(numpy.float32),
            spec=spec.astype(numpy.float32),
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            f0_data=input.f0,
            phoneme_data=input.phoneme,
            spec_data=input.spec,
            silence_data=input.silence,
            phoneme_list_data=input.phoneme_list,
            volume_data=input.volume,
            prepost_silence_length=self.prepost_silence_length,
            max_sampling_length=self.max_sampling_length,
            f0_process_mode=self.f0_process_mode,
            time_mask_max_second=self.time_mask_max_second,
            time_mask_rate=self.time_mask_rate,
        )


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, speaker_ids: List[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.int64)
        return d


class UnbalancedSpeakerFeatureDataset(SpeakerFeatureDataset):
    def __init__(
        self,
        dataset: FeatureDataset,
        speaker_ids: List[int],
        weighted_speaker_id: int,
        weight: int,
    ):
        super().__init__(dataset=dataset, speaker_ids=speaker_ids)

        self.weighted_indexes = [
            i
            for i, speaker_id in enumerate(speaker_ids)
            if speaker_id == weighted_speaker_id
        ]
        self.weight = weight

        assert len(self.weighted_indexes) > 0

    def __len__(self):
        return super().__len__() + len(self.weighted_indexes) * (self.weight - 1)

    def __getitem__(self, i):
        if i >= super().__len__():
            i = self.weighted_indexes[
                (i - super().__len__()) % len(self.weighted_indexes)
            ]
        return super().__getitem__(i)


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create_dataset(config: DatasetConfig):
    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_glob)}
    assert set(fn_list) == set(phoneme_paths.keys())

    spec_paths = {Path(p).stem: Path(p) for p in glob(config.spec_glob)}
    assert set(fn_list) == set(spec_paths.keys())

    silence_paths = {Path(p).stem: Path(p) for p in glob(config.silence_glob)}
    assert set(fn_list) == set(silence_paths.keys())

    phoneme_list_paths: Optional[Dict[str, Path]] = None
    if config.phoneme_list_glob is not None:
        phoneme_list_paths = {
            Path(p).stem: Path(p) for p in glob(config.phoneme_list_glob)
        }
        fn_list = sorted(phoneme_list_paths.keys())
        assert len(fn_list) > 0

    volume_paths: Optional[Dict[str, Path]] = None
    if config.volume_glob is not None:
        volume_paths = {Path(p).stem: Path(p) for p in glob(config.volume_glob)}
        fn_list = sorted(volume_paths.keys())
        assert len(fn_list) > 0

    speaker_ids: Optional[Dict[str, int]] = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.num_speaker == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_test=False):
        inputs = [
            LazyInput(
                f0_path=f0_paths[fn],
                phoneme_path=phoneme_paths[fn],
                spec_path=spec_paths[fn],
                silence_path=silence_paths[fn],
                phoneme_list_path=(
                    phoneme_list_paths[fn] if phoneme_list_paths is not None else None
                ),
                volume_path=volume_paths[fn] if volume_paths is not None else None,
            )
            for fn in fns
        ]

        dataset = FeatureDataset(
            inputs=inputs,
            prepost_silence_length=config.prepost_silence_length,
            max_sampling_length=config.max_sampling_length,
            f0_process_mode=F0ProcessMode(config.f0_process_mode),
            time_mask_max_second=(config.time_mask_max_second if not for_test else 0),
            time_mask_rate=(config.time_mask_rate if not for_test else 0),
        )

        if speaker_ids is not None:
            if config.weighted_speaker_id is None or config.speaker_weight is None:
                dataset = SpeakerFeatureDataset(
                    dataset=dataset,
                    speaker_ids=[speaker_ids[fn] for fn in fns],
                )
            else:
                dataset = UnbalancedSpeakerFeatureDataset(
                    dataset=dataset,
                    speaker_ids=[speaker_ids[fn] for fn in fns],
                    weighted_speaker_id=config.weighted_speaker_id,
                    weight=config.speaker_weight,
                )

        dataset = TensorWrapperDataset(dataset)

        if for_test:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    valid_dataset = (
        create_validation_dataset(config) if config.valid_num is not None else None
    )

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_test=True),
        "valid": valid_dataset,
    }


def create_validation_dataset(config: DatasetConfig):
    assert config.valid_f0_glob is not None
    assert config.valid_phoneme_glob is not None
    assert config.valid_spec_glob is not None
    assert config.valid_silence_glob is not None
    assert config.valid_trial_num is not None

    f0_paths = {Path(p).stem: Path(p) for p in glob(config.valid_f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = {Path(p).stem: Path(p) for p in glob(config.valid_phoneme_glob)}
    assert set(fn_list) == set(phoneme_paths.keys())

    spec_paths = {Path(p).stem: Path(p) for p in glob(config.valid_spec_glob)}
    assert set(fn_list) == set(spec_paths.keys())

    silence_paths = {Path(p).stem: Path(p) for p in glob(config.valid_silence_glob)}
    assert set(fn_list) == set(silence_paths.keys())

    phoneme_list_paths: Optional[Dict[str, Path]] = None
    if config.valid_phoneme_list_glob is not None:
        phoneme_list_paths = {
            Path(p).stem: Path(p) for p in glob(config.valid_phoneme_list_glob)
        }
        fn_list = sorted(phoneme_list_paths.keys())
        assert len(fn_list) > 0

    volume_paths: Optional[Dict[str, Path]] = None
    if config.valid_volume_glob is not None:
        volume_paths = {Path(p).stem: Path(p) for p in glob(config.valid_volume_glob)}
        fn_list = sorted(volume_paths.keys())
        assert len(fn_list) > 0

    speaker_ids: Optional[Dict[str, int]] = None
    if config.valid_speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.valid_speaker_dict_path.read_text()
        )

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    valids = fn_list[: config.valid_num]

    inputs = [
        LazyInput(
            f0_path=f0_paths[fn],
            phoneme_path=phoneme_paths[fn],
            spec_path=spec_paths[fn],
            silence_path=silence_paths[fn],
            phoneme_list_path=(
                phoneme_list_paths[fn] if phoneme_list_paths is not None else None
            ),
            volume_path=volume_paths[fn] if volume_paths is not None else None,
        )
        for fn in valids
    ]

    dataset = FeatureDataset(
        inputs=inputs,
        prepost_silence_length=config.prepost_silence_length,
        max_sampling_length=config.max_sampling_length,
        f0_process_mode=F0ProcessMode(config.f0_process_mode),
        time_mask_max_second=0,
        time_mask_rate=0,
    )

    if speaker_ids is not None:
        if config.weighted_speaker_id is None or config.speaker_weight is None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in valids],
            )
        else:
            dataset = UnbalancedSpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in valids],
                weighted_speaker_id=config.weighted_speaker_id,
                weight=config.speaker_weight,
            )

    dataset = TensorWrapperDataset(dataset)
    dataset = ConcatDataset([dataset] * config.valid_trial_num)
    return dataset
