import librosa
import librosa.display

data, sampling_rate = librosa.load('testAudio.wav')

import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()
