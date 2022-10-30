import scipy.io.wavfile as wav
import numpy as np
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
        file_name: str,
        Fs: np.float,
        time: np.float,
        delay: np.float,
        amp: np.float, 
    ):
        self.phone_number = phone_number
        self.file_name = file_name
        self.Fs = Fs
        self.time = time
        self.delay = delay
        self.amp = amp
        self.signal = self.compose()

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

    def compose(self) -> np.array:
        signal = np.array([])
        for number in self.phone_number:
            tone_delay_signal = self.__dtmf_function(number, self.Fs, self.time, self.delay, self.amp)
            signal = np.append(signal, tone_delay_signal)

        wav.write(self.file_name, self.Fs, signal)
        return signal