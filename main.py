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
from PyQt6.QtWidgets import (
    QApplication, 
    QGridLayout,
    QLineEdit,
    QMainWindow,  
    QPushButton,
    QVBoxLayout,
    QWidget,
)
ERROR_MSG = "ERROR" #错误信息
WINDOW_SIZE = 235#创建了一个Python 常量,固定窗口大小
DISPLAY_HEIGHT = 35#定义显示高度
BUTTON_SIZE = 40


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

def evaluateExpression(expression):
    '''计算表达式'''
    try:
        result = expression
    except Exception:
        result = ERROR_MSG
    return result      #评估成功，则返回result  

class PyCalcWindow(QMainWindow):#此类继承自QMainWindow.
    '''计算器的main window'''

    def __init__(self):#定义了类初始化器
        super().__init__()#超类进行初始化
        self.setWindowTitle("DTMF信号生成与识别")
        self.setFixedSize(WINDOW_SIZE, WINDOW_SIZE)
        self.generalLayout = QVBoxLayout()
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
        self.setCentralWidget(centralWidget)
        self._createDisplay()
        self._createButtons()
        #QWidget对象并将其设置为窗口的中心部件。
        #该对象将是计算器应用程序中所有必需的 GUI 组件的父级。

    def _createDisplay(self):
        self.display = QLineEdit()
        self.display.setFixedHeight(DISPLAY_HEIGHT)
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight)#显示文本左对齐
        self.display.setReadOnly(True)#只读
        self.generalLayout.addWidget(self.display)#显示添加到计算器的总体布局中
   
    def _createButtons(self):
        self.buttonMap = {}#创建空字典来保存计算器按钮
        buttonsLayout = QGridLayout()
        keyBoard = [
            ["7", "8", "9", "A", "C"],
            ["4", "5", "6", "B", "D"],
            ["1", "2", "3", "随机", "清零"],
            ["0", "*", "#", "确认", "播放"],
        ]#创建列表来存储键标签

        for row, keys in enumerate(keyBoard):#外循环遍历行
            for col, key in enumerate(keys):#内循环遍历列
                self.buttonMap[key] = QPushButton(key)
                self.buttonMap[key].setFixedSize(BUTTON_SIZE, BUTTON_SIZE)#按钮都有固定大小的40x40像素
                buttonsLayout.addWidget(self.buttonMap[key], row, col)#创建按钮并将它们添加到self.buttonMap和buttonsLayout中

        self.generalLayout.addLayout(buttonsLayout)#调用对象将网格布局嵌入到计算器的总体布局
    def setDisplayText(self, text):
        '''设置显示的文本'''
        self.display.setText(text)#设置和更新显示的文本
        self.display.setFocus()#设置光标在显示器上的焦点

    def displayText(self): #返回显示的当前文本
        '''获取显示的文本'''
        return self.display.text() 

    def clearDisplay(self):
        '''清除显示'''
        self.setDisplayText("")#显示的文本设置为空字符串 ("")

def evaluateExpression(expression):
    '''计算表达式'''
    try:
        result = expression
    except Exception:
        result = ERROR_MSG
    return result      #评估成功，则返回result  


class PyCalc:
    '''
    访问 GUI 的公共界面。
    处理数学表达式的创建。
    将所有按钮的.clicked信号连接到相应的插槽。'''
    def __init__(self, model, view, dtmt, sound_name):
        self._evaluate = model 
        self._view = view
        self._dtmt = dtmt
        self._sound_name = sound_name
        self._connectSignalsAndSlots()#建立信号和插槽的所有必需连接

    def _calculateResult(self):
        result = self._evaluate(expression=self._view.displayText()) #利用eval()计算,同时displaytext会输入当前表达式
        a = self._dtmt(result)

        self._view.setDisplayText('结果: ' + a)#使用计算结果更新显示文本,即显示运算结果
        return result

    def _buildExpression(self, subExpression): #构建数学表达式
        if self._view.displayText() == ERROR_MSG:
            self._view.clearDisplay()
        expression = self._view.displayText() + subExpression #显示的返回值+输入的subExpression
        self._view.setDisplayText(expression) #显示expression

    def _play_signal(self,name):
        #在这里call外部的play_signal函数
        playsound(name) 
            

    def _connectSignalsAndSlots(self):#将所有按钮的.clicked信号与控制器类中的相应插槽方法连接
        for keySymbol, button in self._view.buttonMap.items():
            if keySymbol not in {"=", "确认","播放","清零"}:
                button.clicked.connect(
                    partial(self._buildExpression, keySymbol)#（call function + 输入的参数）实现数学表达式的构建
                )
        # self._view.buttonMap["="].clicked.connect(self._calculateResult)#若‘=’, call _calculate
        self._view.buttonMap["播放"].clicked.connect(lambda: self._play_signal(self._sound_name))
        self._view.buttonMap["确认"].clicked.connect(self._calculateResult)#若‘确认’, call function
        self._view.display.returnPressed.connect(self._calculateResult)
        # self._view.buttonMap["C"].clicked.connect(self._view.clearDisplay)#若‘C’, call clearDisplay
        self._view.buttonMap["清零"].clicked.connect(self._view.clearDisplay)#若‘C’, call clearDisplay
        #self._view.buttonMap["播放"].clicked.connect(lambda: self._play_signal(self._sound_name))



def read_input(strings):

    # phonenumber = str('1234567A')
    phonenumber = strings 
    samplefrequency = np.int(440000) #生成采样率40k
    toneduration = np.float(0.10)
    delay = np.float(0.10)
    amplitude = np.float(2) #幅度为2 
    out_name = str('haha'+'.wav')
    signal  = DtmfGenerator(phonenumber, out_name, samplefrequency, toneduration, delay, amplitude) #示例化DTMF模拟信号类
    a = signal.compose()   #调用类的方法，生成dtmf模拟信号，存储在a的变量中
    a_awgn = awgn(a,0,1)   #  通过0db的awgn信道

    #samplefrequency = np.int(8000)
    sam_fre = 8000 #伪模拟信号的采样频率
    sig_lasting_time = len(a_awgn) / samplefrequency  #信号持续时间#
    fft_calc = Create_Fft_Sig(sig_lasting_time, sam_fre, a_awgn)
    x_sig = fft_calc.sam_signal() #产生降采样信号

    (freqs, xfa) = caculate_fft(sam_fre,x_sig)
    plot_views = Plot(a_awgn, x_sig, sig_lasting_time, sam_fre, freqs, xfa)
    plot_views.plot_all()
    detect_chr = Find_Character(x_sig, sam_fre)
    print(detect_chr.detect_sig()) 
    return str(detect_chr.detect_sig())

def main():

    name= 'haha'+'.wav'
    '''Py计算器的主函数'''
    pycalcApp = QApplication([])#创建了一个QApplication名为pycalcApp的对象。
    pycalcWindow = PyCalcWindow()#创建应用程序窗口的实例
    pycalcWindow.show()#通过调用窗口对象显示了 GUI 
    PyCalc(model=evaluateExpression, view=pycalcWindow,dtmt=read_input,sound_name=name)
    sys.exit(pycalcApp.exec())
    
if __name__ == "__main__":
    main()        
