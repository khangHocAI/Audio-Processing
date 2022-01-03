import glob
import librosa
import random
import numpy as np
import pretty_midi as pm

class PianoRollDataModule():
    def __init__(self):
        self.train_data = glob.glob("features/*.npz")
        self.valid_data = glob.glob("validation/*.npz")
        self.test_data = glob.glob("dataset/test/*.npz")
        self.prepare_train()
        self.prepare_valid()
        # self.prepare_test()

    def prepare_train(self):
        random.shuffle(self.train_data)
        self.loaded_train_data = np.load(self.train_data[0])
        self.num_train_files = len(self.train_data)-1
        self.file_train_index = 0
        self.current_train_batch_index = 0
    
    def prepare_valid(self):
        random.shuffle(self.valid_data)
        self.loaded_valid_data = np.load(self.valid_data[0])
        self.num_valid_files = len(self.valid_data)-1
        self.file_valid_index = 0
        self.current_valid_batch_index = 0
    
    def prepare_test(self):
        self.loaded_test_data = np.load(self.test_data[0])
        self.num_test_files = len(self.test_data)-1
        self.file_test_index = 0
        self.current_test_batch_index = 0
    
    def get_train_data(self,window_size=5210, midi_size=10,batch_size=64):
        end_epoch = False
        if self.file_train_index == self.num_train_files: #if go through all the files -> reset dataset
            self.prepare_train() # reset dataset
            end_epoch = True
        loaded_audio = self.loaded_train_data['audio']
        loaded_pianoroll = self.loaded_train_data['pianoroll']
        max_batch = min((loaded_audio.shape[0]/window_size)-1, (loaded_pianoroll.shape[1]/midi_size)-1)
        audio = np.zeros([batch_size, window_size])
        piano = np.zeros([batch_size,88,midi_size])
        for i in range (batch_size):
            self.current_train_batch_index += 1
            if self.current_train_batch_index > max_batch:
                if self.file_train_index > self.num_train_files:
                    break
                self.file_train_index+=1
                loaded_data = np.load(self.train_data[self.file_train_index])
                self.current_train_batch_index = 0
                loaded_audio = loaded_data['audio']
                loaded_pianoroll = loaded_data['pianoroll']
                max_batch =min((loaded_audio.shape[0]/window_size)-1, (loaded_pianoroll.shape[1]/midi_size)-1)
            
        return audio,piano, end_epoch
    
    def get_valid_data(self,window_size=5210, midi_size=10,batch_size=64):
        end_epoch = False
        if self.file_valid_index == self.num_valid_files: #if go through all the files -> reset dataset
            self.prepare_valid() # reset dataset
            end_epoch = True
        loaded_audio = self.loaded_valid_data['audio']
        loaded_pianoroll = self.loaded_valid_data['pianoroll']
        max_batch =min((loaded_audio.shape[0]/window_size)-1, (loaded_pianoroll.shape[1]/midi_size)-1)
        audio = np.zeros([batch_size, window_size])
        piano = np.zeros([batch_size,88,midi_size])
        for i in range (batch_size):
            self.current_valid_batch_index += 1
            if self.current_valid_batch_index > max_batch:
                if self.file_valid_index > self.num_valid_files:
                    break
                self.file_valid_index+=1
                loaded_data = np.load(self.valid_data[self.file_valid_index])
                self.current_valid_batch_index = 0
                loaded_audio = loaded_data['audio']
                loaded_pianoroll = loaded_data['pianoroll']
                max_batch =min((loaded_audio.shape[0]/window_size)-1, (loaded_pianoroll.shape[1]/midi_size)-1)
            audio[i,:] = loaded_audio[self.current_valid_batch_index*window_size:self.current_valid_batch_index*window_size+window_size]
            piano[i,:] = loaded_pianoroll[:,self.current_valid_batch_index*midi_size:self.current_valid_batch_index*midi_size+midi_size]
        return audio,piano, end_epoch
