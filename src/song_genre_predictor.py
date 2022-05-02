import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from tensorflow.keras.models import load_model
import math
import sys
import librosa


def mfcc_extractor(audio_file,track_duration):
    SAMPLE_RATE=22050
    NUM_MFCC=13
    HOP_LENGTH=512
    N_FFT=2048
    TRACK_DURATION=track_duration
    SAMPLE_PER_TRACK=SAMPLE_RATE*TRACK_DURATION
    NUM_SEGMENT=5
    samples_per_segment=int(SAMPLE_PER_TRACK/NUM_SEGMENT)
    num_mfcc_vectors_per_segment=math.ceil(samples_per_segment/HOP_LENGTH)
    
    signal,sr=librosa.load(audio_file,sr=SAMPLE_RATE)
    
    for d in range(NUM_SEGMENT):
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        # return only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            return mfcc


if len(sys.argv)!=2:
    print("please give audio file name as argument")
    exit(1)
    
model=load_model('./cnn_model.h5')

audio_file=sys.argv[1]
with open('./genres_mapping.json','r') as f:
    music_genre=json.load(f)['mapping']

mfcc=mfcc_extractor(audio_file,30)

mfcc=mfcc[np.newaxis,...,np.newaxis]
predicted_result=model.predict(mfcc)

predicted_index=np.argmax(predicted_result)

print("genre of the song {} is {}".format(audio_file.split('\\')[-1],music_genre[predicted_index]))