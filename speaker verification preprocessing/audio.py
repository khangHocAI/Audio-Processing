import logging
import os
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
from python_speech_features import fbank
from tqdm import tqdm

from preprocessing.constants import SAMPLE_RATE, NUM_FBANKS
from preprocessing.utils import find_files, ensures_dir
from preprocessing.unsupervised_vad import remove_silence
logger = logging.getLogger(__name__)
 

def read_mfcc(input_filename, sample_rate=SAMPLE_RATE):
    import matplotlib.pyplot as plt
    audio = Audio.read(input_filename, sample_rate)
    audio_voice_only = remove_silence(audio,sample_rate)
    # from scipy.io.wavfile import write
    # write('test.wav', 16000, audio_voice_only)
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc

def extract_speaker_and_utterance_ids(filename: str):  # LIBRI.
    # 'audio/dev-other/116/288045/116-288045-0000.flac'
    speaker, _, basename = Path(filename).parts[-3:]
    filename.split('-')
    utterance = os.path.splitext(basename.split('-', 1)[-1])[0]
    assert basename.split('-')[0] == speaker
    return speaker, utterance

class Audio:

    def __init__(self, cache_dir: str, audio_dir: str = None, sample_rate: int = SAMPLE_RATE, ext='flac'):
        self.ext = ext
        self.cache_dir = os.path.join(cache_dir, 'audio-fbanks')
        ensures_dir(self.cache_dir)
        if audio_dir is not None:
            self.build_cache(os.path.expanduser(audio_dir), sample_rate)
        self.speakers_to_utterances = defaultdict(dict)
        for cache_file in find_files(self.cache_dir, ext='npy'):
            # /path/to/speaker_utterance.npy
            speaker_id, utterance_id = Path(cache_file).stem.split('_')
            self.speakers_to_utterances[speaker_id][utterance_id] = cache_file

    @property
    def speaker_ids(self):
        return sorted(self.speakers_to_utterances)

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f'audio_dir: {audio_dir}.')
        logger.info(f'sample_rate: {sample_rate:,} hz.')
        audio_files = find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        with tqdm(audio_files) as bar:
            for audio_filename in bar:
                bar.set_description(audio_filename)
                self.cache_audio_file(audio_filename, sample_rate)

    def cache_audio_file(self, input_filename, sample_rate):
        sp, utt = extract_speaker_and_utterance_ids(input_filename)
        cache_filename = os.path.join(self.cache_dir, f'{sp}_{utt}.npy')
        if not os.path.isfile(cache_filename):
            try:
                mfcc = read_mfcc(input_filename, sample_rate)
                np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)


def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc


def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, _ = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS, winfunc=np.hamming)
    frames_features = normalize_frames(filter_banks)
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.

def normalize_3_seconds(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]

def normalize_frames(m, epsilon=1e-12):
    #normalize for each 300 frames (3 seconds)
    num_frames = m.shape[0]
    loop = int(num_frames/300)
    pos = 0
    for _ in range (loop):
        m[pos:pos+299,:] = normalize_3_seconds(m[pos:pos+299,:])
        pos+=300
    if pos < num_frames:
        m[pos:num_frames,:] = normalize_3_seconds(m[pos:num_frames,:])
    return m
    

    