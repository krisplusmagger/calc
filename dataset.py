import numpy as np
from scipy.fft import fft
import scipy.io.wavfile as wav
from commpy.channels import awgn
from plot import Plot
from findcharacter import Find_Character
from demfgenerator import DtmfGenerator

INPUT_TABLE = ["1", "2", "3", "A", "4", "5", "6", "B", "7", "8", "9", "C", "*", "0", "#", "D"]

def generate_input(num):
    #np.random.seed(123)  # 设置种子为123
    random_number = np.random.randint(low=0, high=4)  # 生成0到3之间的随机整数
    input_list = []
    for i in range(num):
        #random_number = np.random.randint(low=0, high=16)
        input_list.append(INPUT_TABLE[i])
    return input_list

def dtmf(input_list):
    sample_frequency = 128000
    toneduration = 0.01
    delay = 0.0
    #amplitude = 2.0
    dtmf_signals = []

    for input_num in input_list:
        for snr in range(-18, 17):#对应-18-16dB的信噪比
            for j in range(100):    ##每个标签值有1000段数据，每段数据长度为128个点
                amplitude = np.random.uniform(low=0.5, high=3.0)
                dtmf_signal = DtmfGenerator(input_num, "haha", sample_frequency, toneduration, delay, amplitude).compose()
                dtmf_signal_awgn = awgn(dtmf_signal, snr, 1)
                dtmf_signals.append(input_num) #每段的第一点是对应的标签值
                dtmf_signals.append(str(snr)) #每段的第二点是对应的信噪比
                dtmf_signals.append(dtmf_signal_awgn)
    return dtmf_signals

def main():
    #信噪比范围为-18-16dB
    dataset = []
    input_list = generate_input(16)
    dtmf_signals = dtmf(input_list)
    # 保存为.npy文件
    np.save('dtmf_signals.npy', dtmf_signals)



if __name__ == "__main__":
    main()
