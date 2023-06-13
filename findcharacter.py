import numpy as np
from scipy.fft import fft
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


class Find_Character():
    def __init__(self,sampled_signal,sam_freq): 
        self.sampled_signal = sampled_signal
        self.sam_freq = sam_freq

    def caculate_fft(self,sam_freq, sampled_signal):
        # 根据采样率和采样长度计算出每个下标对应的真正频率,频率轴
        freqs = np.linspace(0,sam_freq/2, len(sampled_signal)//2 + 1)
        # wave = np.fft.fft(signal)
        # 利用np.fft.rfft()进行FFT计算
        #代码中将xf除以len(sampled_signal)。这是因为在傅里叶变换中，归一化是常见的操作，通过除以序列的长度可以确保结果的幅度正确表示原始信号。
        xf = np.fft.rfft(sampled_signal) / len(sampled_signal)
        # 取绝对值表示幅值
        #wavea = np.abs(wave)
        xfa = np.abs(xf)
        return freqs, xfa
        
    def judge_char(self, high_freq, low_freq):
        ''' 判断字符 '''
        delta = 10  # 频率识别允许误差半径
        for row in range(len(FREQ_ROW)):
            for col in range(len(FREQ_COL)):
                row_check = FREQ_ROW[row]-delta < high_freq < FREQ_ROW[row]+delta #找到了返回true
                col_check = FREQ_COL[col]-delta < low_freq < FREQ_COL[col]+delta ##找到了返回true
                if row_check and col_check:
                    p = FREQ_TABLE[(FREQ_ROW[row], FREQ_COL[col])]
                    return p
        return None

    def detect_one(self,data,sam_fre):
        ''' 识别一位号码 '''
        # 计算频谱
        (freqs, xfa) = self.caculate_fft(sam_fre,data)
        # 寻找高频和低频
        local = []
        for i in np.arange(1, len(xfa)-1):
            if xfa[i] > xfa[i-1] and xfa[i] > xfa[i+1]:
                local.append(xfa[i])
        local = sorted(local)
        if(len(local) < 2):
            return None
        loc = np.where(xfa == local[-1])
        high_freq = freqs[loc[0][0]]
        loc = np.where(xfa == local[-2])
        low_freq = freqs[loc[0][0]]
        temp = 0
        if low_freq > high_freq:
            temp = high_freq
            high_freq = low_freq
            low_freq = temp

        # 判断字符
        p = self.judge_char(high_freq, low_freq)
        if(p != None):
            # 查看具体频率
            print('high_freq, low_freq = {:.2f}, {:.2f}'.format(high_freq, low_freq))
        return p    
    #监测输入信号并识别数字
    def detect_sig(self):
        length = len(self.sampled_signal)
        result = []  # result
        digit = []  # signal that needed to be processed 
        # 通过采样率除以最低频率向上取整计算得出

        interval = int(np.ceil(self.sam_freq / min(min(FREQ_ROW), min(FREQ_COL)))) #间隔
        # initiate threshold
        # by (mean + (max-mini)* proportion)
        threshold = int(np.mean(self.sampled_signal) + (np.max(self.sampled_signal) - np.mean(self.sampled_signal))*0.2)
        for i in range(length - interval):
            # 根据阈值和间隔判断是否是有效信号
            if np.max([self.sampled_signal[i] for i in range(i, i + interval)]) > threshold:
                digit.append(self.sampled_signal[i]) #利用threshold记录有效区间
            else:
                if len(digit) != 0:
                    p = self.detect_one(digit,self.sam_freq) 
                    if(p != None):  
                        result.append(p)
                    digit = []
        return result 
    
    def detect_signal_2(self):
        length = len(self.sampled_signal)
        result = []  # result
        digit = []  # signal that needed to be processed 
        # 通过采样率除以最低频率向上取整计算得出
        value_time = 0
        interval = int(np.ceil(self.sam_freq / min(min(FREQ_ROW), min(FREQ_COL)))) #间隔
        # initiate threshold
        # by (mean + (max-mini)* proportion)
        #threshold = int(np.mean(self.sampled_signal) + (np.max(self.sampled_signal) - np.mean(self.sampled_signal))*0.2)
        for i in range(length - interval):
                
                for i in range(i, i + interval):
                    digit.append(self.sampled_signal[i]) #记录每段区间 
                else:
                    if len(digit) != 0:
                        p = self.detect_one(digit,self.sam_freq) 
                        if(p != None):  
                            result.append(p)
                        digit = []
        return result 


