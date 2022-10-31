import numpy as np
import matplotlib.pyplot as plt

class Plot():

    def __init__(self,signal,sampled_signal, lasting_time, sampleing_freq,freqs,fft_sig):
        self.signal = signal
        self.sampled_signal = sampled_signal
        self.sampling_freq = sampleing_freq
        self.lasting_time = lasting_time
        self.freqs = freqs 
        self.fft_sig = fft_sig 
        
    def plot_origin_signal(self):
        #信号持续时间：序列长度/产生的频率
        endingtime = len(self.signal) / self.sampling_freq
        #a 是原始的时域信号
        # len(numbers) * toneduration * delay + 0.5
        # 根据采样率和采样长度生成时间序列
        time = np.linspace(0,endingtime,len(self.signal),endpoint=False)
        plt.figure(dpi=200)
        plt.plot(time, self.signal, linewidth = 0.05)
        plt.xlim(0,int(np.ceil(endingtime)))
        plt.xlabel("time (second)")
        plt.title('Original Signal in Time Domain')
        plt.show()
        #linspace（开始，停止， num = 50）num 样本数
        #arange 允许定义步长的大小。linspace 允许定义步数

    def plot_sam_signal(self):
        time = np.linspace(0,self.lasting_time,len(self.signal),endpoint=False)
        plt.figure(dpi=200)
        plt.plot(time, self.signal, linewidth = 0.05)
        plt.xlim(0,int(np.ceil(self.lasting_time)))
        plt.xlabel("time (second)")
        plt.title('Sampling Signal in Time Domain')
        plt.show()

    def plot_fft(self):
        plt.figure(dpi=200)
        plt.title("Freq Domain")
        plt.xlabel("freq (Hz)")
        plt.ylabel('A')
        plt.xlim(0,1500)
        plt.plot(self.freqs[1:], self.fft_sig[1:], linewidth = 0.5)  # [1:]去掉f=0的直流信号
        plt.show()


    def plot_all(self):
        self.plot_origin_signal()
        self.plot_sam_signal()
        self.plot_fft()    