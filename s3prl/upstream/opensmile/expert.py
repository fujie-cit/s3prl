from typing import List, Union, Dict
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import opensmile

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000

class UpstreamExpert(UpstreamBase):
    def __init__(self, feature_set, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, feature_set),
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        self.downsample_rate = 10 * SAMPLE_RATE / 1000

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def forward(self, wavs):
        # import ipdb; ipdb.set_trace()
        feats = []
        device = 'cpu'
        if len(wavs) > 0:
            device = wavs[0].device
        for wav in wavs:
            result = self.smile.process_signal(wav.cpu(), SAMPLE_RATE)
            result = result.to_numpy()
            if result.shape[0] != 0 and result.shape[0] < len(wav) // self.downsample_rate - 2:
                x = np.repeat(result[:1, :], (len(wav) // self.downsample_rate - 2) - result.shape[0], axis=0)
                result = np.concatenate([x, result], axis=0)
                # import ipdb; ipdb.set_trace()
            feats.append(torch.tensor(result))
        padded_feats = pad_sequence(feats, batch_first=True)
        padded_feats = padded_feats.to(device)
        return {
            "last_hidden_state": padded_feats,
            "hidden_states": [padded_feats]
        }
