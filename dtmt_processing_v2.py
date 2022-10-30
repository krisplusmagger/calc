import sys
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

class DtmfGenerator:
    #按键组合DTMF_TABLE
    DTMF_TABLE = {
        "1": np.array([1209, 697]),
        "2": np.array([1336, 697]),
        "3": np.array([1477, 697]),
        "A": np.array([1633, 697]),
        "4": np.array([1209, 770]),
        "5": np.array([1336, 770]),
        "6": np.array([1477, 770]),
        "B": np.array([1633, 770]),
        "7": np.array([1209, 852]),
        "8": np.array([1336, 852]),
        "9": np.array([1477, 852]),
        "C": np.array([1633, 852]),
        "*": np.array([1209, 941]),
        "0": np.array([1336, 941]),
        "#": np.array([1477, 941]),
        "D": np.array([1633, 941]),
    }

    def __init__(
        self,
        phone_number: str,
        Fs: np.float,
        time: np.float,
        delay: np.float,
        amp: np.float, 
    ):
        self.signal = self.compose(phone_number, Fs, time, delay, amp)

    def __dtmf_function(
        self, 
        number: str, 
        Fs: np.float, 
        time: np.float, 
        delay: np.float, 
        amp: np.float,
    ) -> np.array:

        time_tone = np.arange(0, time, (1 / Fs))
        time_delay = np.arange(0, delay, (1 / Fs))

        tone_samples = amp * (
            np.sin(2 * np.pi * self.DTMF_TABLE[number][0] * time_tone)
            + np.sin(2 * np.pi * self.DTMF_TABLE[number][1] * time_tone)
        ) #声音样本(高频分量加低频分量)
        delay_samples = np.sin(2 * np.pi * 0 * time_delay)

        return np.append(tone_samples, delay_samples)

    def compose(
        self,
        phone_number: str,
        Fs: np.float,
        time: np.float,
        delay: np.float,
        amp: np.float,
    ) -> np.array:
        signal = np.array([])

        for number in phone_number:
            tone_delay_signal = self.__dtmf_function(number, Fs, time, delay, amp)
            signal = np.append(signal, tone_delay_signal)

        return signal


def dtmf_tone_generator(
    nums: str,
    samfreq: int,
    toneduration: np.float,
    delay_time: np.float,
    amp: np.float,
    file_name: str,
    ):
    dtmf = DtmfGenerator( 
    nums,
    samfreq,
    toneduration,
    delay_time,
    amp,
    )
    wav.write(file_name, samfreq, dtmf.signal)
    return dtmf.signal 

class 


def plot_origin_signal(signal, sampling_freq):
    #信号持续时间：序列长度/产生的频率
    endingtime = len(signal) / sampling_freq
    #a 是原始的时域信号
    # len(numbers) * toneduration * delay + 0.5
    # 根据采样率和采样长度生成时间序列
    time = np.linspace(0,endingtime,len(signal),endpoint=False)
    plt.figure(dpi=200)
    plt.plot(time, signal, linewidth = 0.05)
    plt.xlim(0,int(np.ceil(endingtime)))
    plt.xlabel("time (second)")
    plt.title('Original Signal in Time Domain')
    plt.show()
    #linspace（开始，停止， num = 50）num 样本数
    #arange 允许定义步长的大小。linspace 允许定义步数

'''模拟采样过程
所有的点数/信号持续时间/模拟采样频率 = 间隔点数'''

def sam_signal(lasting_time,sam_freq,signal):
    sampling_signal = np.array([])
    the_space = int(len(signal)/lasting_time/sam_freq)
    sampling_signal = signal[::the_space]
    return sampling_signal

def plot_sam_signal(lasting_time, signal):

    time = np.linspace(0,lasting_time,len(signal),endpoint=False)
    plt.figure(dpi=200)
    plt.plot(time, signal, linewidth = 0.05)
    plt.xlim(0,int(np.ceil(lasting_time)))
    plt.xlabel("time (second)")
    plt.title('Sampling Signal in Time Domain')
    plt.show()


def caculate_fft(signal, sampling_freq):
    # 根据采样率和采样长度计算出每个下标对应的真正频率,频率轴
    freqs = np.linspace(0, sampling_freq/2, len(signal)//2 + 1)
    # wave = np.fft.fft(signal)
    # 利用np.fft.rfft()进行FFT计算
    xf = np.fft.rfft(signal) / len(signal)
    # 取绝对值表示幅值
    #wavea = np.abs(wave)
    xfa = np.abs(xf)
    return freqs, xfa

def plot_fft(freqs, xfa):
    plt.figure(dpi=200)
    plt.title("Freq Domain")
    plt.xlabel("freq (Hz)")
    plt.ylabel('A')
    plt.xlim(0,1500)
    plt.plot(freqs[1:], xfa[1:], linewidth = 0.5)  # [1:]去掉f=0的直流信号
    plt.show()



def find_freq(freqs, xfa):
    local = []
    for i in np.arange(1, len(xfa)-1):
        if xfa[i] > xfa[i-1] and xfa[i] > xfa[i+1]:
            local.append(xfa[i])
    local = sorted(local)
    loc = np.where(xfa == local[-1])
    high_freq = freqs[loc[0][0]]
    loc = np.where(xfa == local[-2])
    low_freq = freqs[loc[0][0]]     
     # 查看具体频率，调试很有用
    # print('high_freq, low_freq = {:.2f}, {:.2f}'.format(high_freq, low_freq))   
    # print(local)  
    return local 


    '''
    1209 1336 1477 1633
697   1   2   3   A
770   4   5   6   B
852   7   8   9   C
941   *   0   #   D
'''
FREQ_ROW = (1209, 1336, 1477, 1633) #行分布
FREQ_COL = (697, 770, 852, 941)     #列分布
# 频率号码对应表
FREQ_TABLE = {
    (FREQ_ROW[0], FREQ_COL[0]): '1',
    (FREQ_ROW[1], FREQ_COL[0]): '2',
    (FREQ_ROW[2], FREQ_COL[0]): '3',
    (FREQ_ROW[3], FREQ_COL[0]): 'A',

    (FREQ_ROW[0], FREQ_COL[1]): '4',
    (FREQ_ROW[1], FREQ_COL[1]): '5',
    (FREQ_ROW[2], FREQ_COL[1]): '6',
    (FREQ_ROW[3], FREQ_COL[1]): 'B',

    (FREQ_ROW[0], FREQ_COL[2]): '7',
    (FREQ_ROW[1], FREQ_COL[2]): '8',
    (FREQ_ROW[2], FREQ_COL[2]): '9',
    (FREQ_ROW[3], FREQ_COL[2]): 'C',

    (FREQ_ROW[0], FREQ_COL[3]): '*',
    (FREQ_ROW[1], FREQ_COL[3]): '0',
    (FREQ_ROW[2], FREQ_COL[3]): '#',
    (FREQ_ROW[3], FREQ_COL[3]): 'D',
}

def judge_char(high_freq, low_freq):
    ''' 判断字符 '''

    p = ''
    delta = 10  # 频率识别允许误差半径

    for row in range(len(FREQ_ROW)):
        for col in range(len(FREQ_COL)):
            row_check = FREQ_ROW[row]-delta < high_freq < FREQ_ROW[row]+delta #找到了返回true
            col_check = FREQ_COL[col]-delta < low_freq < FREQ_COL[col]+delta ##找到了返回true
            if row_check and col_check:
                p = FREQ_TABLE[(FREQ_ROW[row], FREQ_COL[col])]

    return p

def detect_one(data,sam_fre):
    ''' 识别一位号码 '''

    # 绘制每一位号码的时域和频域图像
    # plot_diagram(data)

    # 计算频谱
    (freqs, xfa) = caculate_fft(data, sam_fre)
    # 寻找高频和低频
    local = []
    for i in np.arange(1, len(xfa)-1):
        if xfa[i] > xfa[i-1] and xfa[i] > xfa[i+1]:
            local.append(xfa[i])
    local = sorted(local)
    loc = np.where(xfa == local[-1])
    high_freq = freqs[loc[0][0]]
    loc = np.where(xfa == local[-2])
    low_freq = freqs[loc[0][0]]
    temp = 0
    if low_freq > high_freq:
        temp = high_freq
        high_freq = low_freq
        low_freq = temp
    # 查看具体频率
    print('high_freq, low_freq = {:.2f}, {:.2f}'.format(high_freq, low_freq))
    # 判断字符
    p = judge_char(high_freq, low_freq)

    return p    

def detect_sig(sam_freq,sampled_signal):
    length = len(sampled_signal)
    result = []  # result
    digit = []  # signal that needed to be processed 
    # 通过采样率除以最低频率向上取整计算得出

    interval = int(np.ceil(sam_freq / min(min(FREQ_ROW), min(FREQ_COL))))
    # initiate threshold
    # by (mean + (max-mini)* proportion)
    threshold = int(np.mean(sampled_signal) + (np.max(sampled_signal) - np.mean(sampled_signal))*0.2)
    for i in range(length - interval):
        # 根据阈值和间隔判断是否是有效信号
        if np.max([sampled_signal[i] for i in range(i, i + interval)]) > threshold:
            digit.append(sampled_signal[i]) #利用threshold记录有效区间
        else:
            if len(digit) != 0:
                p = detect_one(digit,sam_freq) 
                result.append(p)
                digit = []

    return result    

def main():
    
    phonenumber = str(1234567)
    samplefrequency = np.int(400000)
    toneduration = np.float(0.10)
    delay = np.float(0.10)
    amplitude = np.float(2)
    out_name = str('te')
    #信号持续时间为len(numbers) * toneduration * delay + 0.5
    signal_lastingtime = len(phonenumber) * (toneduration + delay) 
    sam_fre = 10000
    a  = dtmf_tone_generator(phonenumber, samplefrequency, toneduration, delay, amplitude, out_name)
    endpalce = len(a) / samplefrequency
    plot_origin_signal(a, samplefrequency)
    
    x_sig = (sam_signal(endpalce,sam_fre, a))
    plot_sam_signal(signal_lastingtime, x_sig)

    (freqs, xfa) = caculate_fft(x_sig,sam_fre)
    plot_fft(freqs, xfa)

    print(detect_sig(sam_fre, x_sig)) 
    


if __name__ == "__main__":
    main()        
