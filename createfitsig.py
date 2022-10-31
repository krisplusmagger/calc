import numpy as np 
class Create_Fft_Sig():

    def __init__(self, lasting_time, sam_freq, signal):
        self.lasting_time = lasting_time
        self.sam_freq = sam_freq
        self.signal = signal 

    '''模拟采样过程
    所有的点数/信号持续时间/模拟采样频率 = 间隔点数'''
    def sam_signal(self):
        sampling_signal = np.array([])
        the_space = int(len(self.signal)/self.lasting_time/self.sam_freq)
        sampling_signal = self.signal[::the_space]
        return sampling_signal