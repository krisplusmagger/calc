import sys
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt 
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from plot import Plot
from findcharacter import Find_Character
from demfgenerator import DtmfGenerator
from createfitsig import Create_Fft_Sig
from pycalc import PyCalcWindow, PyCalc
from unittest import result#导入sys. 该模块提供了exit()用于彻底终止应用程序的功能
##import commpy.channels.awgn as awgn
from commpy.channels import awgn
from PyQt6.QtCore import Qt 
from playsound import playsound
from functools import partial

ERROR_MSG = "ERROR" #错误信息
WINDOW_SIZE = 235#创建了一个Python 常量,固定窗口大小
DISPLAY_HEIGHT = 35#定义显示高度
BUTTON_SIZE = 40

INPUT_TABLE = ["1", "2", "3", "A", "4", "5", "6", "B", "7", "8", "9", "C", "*", "0", "#", "D"]

def caculate_fft(sam_freq, sampled_signal):
    # 根据采样率和采样长度计算出每个下标对应的真正频率,频率轴
    freqs = np.linspace(0,sam_freq/2, len(sampled_signal)//2 + 1)
    # wave = np.fft.fft(signal)
    # 利用np.fft.rfft()进行FFT计算
    xf = np.fft.rfft(sampled_signal) / len(sampled_signal)
    # 取绝对值表示幅值
    #wavea = np.abs(wave)
    xfa = np.abs(xf)
    return freqs, xfa

def generate_random_input(num):
    np.random.seed(123)  # 设置种子为123
    random_number = np.random.randint(low=0, high=16)  # 生成0到15之间的随机整数
    input_list = []
    ##生成128位输入序列
    for i in range(num):
        random_number = np.random.randint(low=0, high=16)
        input_list.append(INPUT_TABLE[random_number])
    return input_list

def dtmf(input_num,snr):
    # phonenumber = str('1234567A')
    phonenumber = input_num
    samplefrequency = np.int(8000) #生成采样率8k，每段800点
    toneduration = np.float(0) #信号持续时间0s
    delay = np.float(0.10)        #空白时间0.1s
    amplitude = np.float(2) #幅度为2 
    out_name = str('haha'+'.wav')
    signal  = DtmfGenerator(phonenumber, out_name, samplefrequency, toneduration, delay, amplitude) #示例化DTMF模拟信号类
    a = signal.compose()   #调用类的方法，生成dtmf模拟信号，存储在a的变量中
    a_awgn = awgn(a,snr,1)   #  通过snr db的awgn信道， snr为输入设定的信噪比
    return a_awgn

def recognition(sam_freq, input_signal):
    
    #sig_lasting_time = len(input_signal) / sam_freq #信号持续时间#
    #(freqs, xfa) = caculate_fft(sam_freq,input_signal)
    #plot_views = Plot(a_awgn, x_sig, sig_lasting_time, sam_freq, freqs, xfa)
    #plot_views.plot_all()
    detect_chr = Find_Character(input_signal, sam_freq)
    results = detect_chr.detect_sig()
    #print(results) #打印识别的结果
    return results

def bit_error(original_list, recognition_output):
    #assert len(original_list) == len(recognition_output), "输入长度必须相等"
    error_bits = 0
    for i in range(24):
        if (original_list[i] != recognition_output[i]):
            error_bits += 1

    return float(error_bits / len(original_list))
    
def main():
    name= 'haha'+'.wav'
    '''Py计算器的主函数'''
    input_list = generate_random_input(24) #生成128位伪随机序列
    dtmf_sig = dtmf(input_list,snr = 0) #生成dtmf信号,信噪比为0dB
    results =  recognition(sam_freq=8000,input_signal = dtmf_sig) #识别信号，并将识别结果存储在results变量中
    error_rates = bit_error(input_list,results)
    print(error_rates)
    
if __name__ == "__main__":
    main()        