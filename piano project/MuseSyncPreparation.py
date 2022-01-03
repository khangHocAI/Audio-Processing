import glob
import numpy as np
import os
import pretty_midi as pm
pm.pretty_midi.MAX_TICK = 1e10
import librosa
def prepare_dataset():
    """Calculate spectrograms, pianoroll and save the pre-calculated features.

            Args:
                metadata: metadata to the dataset.
                spectrogram_setting: the spectrogram setting (type and parameters).
            Returns:
                No return, save pre-calculated features instead.
    """
    flacs = glob.glob("dataset/*.flac")
    print(flacs)
    num_files = len(flacs)
    for i in range(num_files):
        audio_file = flacs[i]
        audio, fs = librosa.load(audio_file, sr=16000)
        midi_file = audio_file.replace("flac","mid")
        midi_data = pm.PrettyMIDI(midi_file)
        pianoroll = midi_data.get_piano_roll(fs=1./0.05)[21:21+88]  # 88 piano keys, hop_time=50ms
        # data = {"spectrogram":spectrogram, "pianoroll":pianoroll}

        file_name = audio_file.replace(".flac","")
        np.savez(f"{file_name}", audio=audio, pianoroll=pianoroll)
prepare_dataset()
