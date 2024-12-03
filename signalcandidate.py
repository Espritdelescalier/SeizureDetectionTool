from dataclasses import dataclass


@dataclass
class SignalCandidate:
    onset: int
    end: int
    capture_modality: str


@dataclass
class MetaAndDataFile:
    sample_rate: int
    signal_length: int
    full_signal: list
    number_channels: int
    eeg_channel_index: int
    channels_labels: list
