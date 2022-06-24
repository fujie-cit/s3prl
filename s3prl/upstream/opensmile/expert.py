from typing import List, Union, Dict
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
            r = self.smile.process_signal(wav.cpu(), SAMPLE_RATE)
            feats.append(torch.tensor(r.to_numpy()))
        padded_feats = pad_sequence(feats, batch_first=True)
        padded_feats = padded_feats.to(device)
        return {
            "last_hidden_state": padded_feats,
            "hidden_states": [padded_feats]
        }
