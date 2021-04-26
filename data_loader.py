import os
import librosa
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from preprocess import (FEATURE_DIM, FFTSIZE, FRAMES, SAMPLE_RATE,
                        world_features)
from utility import Normalizer, speakers
import random

class AudioDataset(Dataset):
    """docstring for AudioDataset."""
    def __init__(self, datadir:str):
        super(AudioDataset, self).__init__()
        self.datadir = datadir
        self.num_speakers = speakers

    def __getitem__(self, idx1):

        self.speaker1 = self.num_speakers[idx1]
        self.files1 = librosa.util.find_files(os.path.join(self.datadir, self.speaker1), ext='npy')
        p1 = self.files1[random.randint(0,len(self.files1)-1)]

        idx2 = random.randint(0, len(self.num_speakers)-1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(self.num_speakers)-1)

        self.speaker2 = self.num_speakers[idx2]
        self.files2 = librosa.util.find_files(os.path.join(self.datadir, self.speaker2), ext='npy')
        p2 = self.files2[random.randint(0,len(self.files2)-1)]

        mcep1 = np.load(p1)
        mcep1 = torch.FloatTensor(mcep1)
        mcep1 = torch.unsqueeze(mcep1, 0)

        mcep2 = np.load(p2)
        mcep2 = torch.FloatTensor(mcep2)
        mcep2 = torch.unsqueeze(mcep2, 0)

        return mcep1, mcep2

    def __len__(self):
        return len(self.num_speakers)

def data_loader(datadir: str, batch_size=8, shuffle=True, num_workers=10):
    '''if mode is train datadir should contains training set which are all npy files
        or, mode is test and datadir should contains only wav files.
    '''
    dataset = AudioDataset(datadir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader



class TestSet(object):
    """docstring for TestSet."""
    def __init__(self, datadir:str):
        super(TestSet, self).__init__()
        self.datadir = datadir
        self.norm = Normalizer()
        
    def choose(self):
        '''choose one speaker for test'''
        r = random.choice(speakers)
        return r
    
    def test_data(self, src_speaker=None, trg_speaker=None):
        '''choose one speaker for conversion'''


        if src_speaker==None:
            src_speaker = self.choose()
        if trg_speaker != None:
            while src_speaker == trg_speaker:
                src_speaker = self.choose()
        if trg_speaker == None:
            trg_speaker = self.choose()
            while src_speaker == trg_speaker:
                trg_speaker = self.choose()

        assert src_speaker != trg_speaker

        wavfiles1 = librosa.util.find_files(os.path.join(self.datadir, src_speaker), ext='wav')

        wavfiles2 = librosa.util.find_files(os.path.join(self.datadir, trg_speaker), ext='wav')

        res1 = {}
        for f in wavfiles1:
            filename = os.path.basename(f)
            wav, _ = librosa.load(f, sr=SAMPLE_RATE, dtype=np.float64)
            f0, timeaxis, sp, ap, coded_sp = world_features(wav, SAMPLE_RATE, FFTSIZE, FEATURE_DIM)
            coded_sp_norm = self.norm.forward_process(coded_sp.T, src_speaker)

            if not res1.__contains__(filename):
                res1[filename] = {}
            res1[filename]['coded_sp_norm'] = np.asarray(coded_sp_norm)
            res1[filename]['f0'] = np.asarray(f0)
            res1[filename]['ap'] = np.asarray(ap)

        trg_wav = wavfiles2[random.randint(0, len(wavfiles2)-1)]

        wav, _ = librosa.load(trg_wav, sr=SAMPLE_RATE, dtype=np.float64)
        f0, timeaxis, sp, ap, coded_sp = world_features(wav, SAMPLE_RATE, FFTSIZE, FEATURE_DIM)
        coded_sp_norm = self.norm.forward_process(coded_sp.T, src_speaker)
        if coded_sp_norm.shape[1] < FRAMES:
            trg_mel = np.pad(coded_sp_norm,((0, FRAMES-coded_sp_norm.shape[1]),(0, 0)))
        else:
            start = random.randint(0, coded_sp_norm.shape[1] - FRAMES - 1)
            trg_mel = coded_sp_norm[:, start:start + FRAMES]
        trg_mel = np.asarray(trg_mel)

        return res1, trg_mel, src_speaker, trg_speaker

if __name__=='__main__':
    pass





