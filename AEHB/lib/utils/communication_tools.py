import numpy as np
import tensorflow as tf
from functools import reduce


class signal:
    """"""
    def __init__(self, mode=None, norm=None, Ns=None):
        mode = mode.upper()
        self.norm = norm
        self.Ns = Ns
        self.BPSK = {"dictionary": {'1': 1 + 0j, '0': -1 + 0j},
                     "scatter": [1, -1]
                     }
        self.QPSK_A = {"dictionary": {'11': 1 + 0j, '10': 0 + 1j, '01': -1 + 0j, '00': 0 - 1j},
                       "scatter": [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
                       }
        self.QPSK_B = {"dictionary": {'11': 1 + 1j, '10': 1 - 1j, '01': -1 + 1j, '00': -1 - 1j},
                       "scatter": [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
                       }
        self.PSK_8 = {"dictionary": {'000': np.exp(1j * np.pi * 0),     '001': np.exp(1j * np.pi * 1 / 4), '011': np.exp(1j * np.pi * 2 / 4), '010': np.exp(1j * np.pi * 3 / 4),
                                     '110': np.exp(1j * np.pi * 4 / 4), '111': np.exp(1j * np.pi * 5 / 4), '101': np.exp(1j * np.pi * 6 / 4), '100': np.exp(1j * np.pi * 7 / 4)},
                      "scatter": [np.exp(1j * np.pi * 0),     np.exp(1j * np.pi * 1 / 4), np.exp(1j * np.pi * 2 / 4), np.exp(1j * np.pi * 3 / 4),
                                  np.exp(1j * np.pi * 4 / 4), np.exp(1j * np.pi * 5 / 4), np.exp(1j * np.pi * 6 / 4), np.exp(1j * np.pi * 7 / 4)]
                      }
        self.QAM_16 = {"dictionary": {'1000': -3 + 3j, '1001': -1 + 3j, '1011': 1 + 3j, '1010': 3 + 3j,
                                      '1100': -3 + 1j, '1101': -1 + 1j, '1111': 1 + 1j, '1110': 3 + 1j,
                                      '0100': -3 - 1j, '0101': -1 - 1j, '0111': 1 - 1j, '0110': 3 - 1j,
                                      '0000': -3 - 3j, '0001': -1 - 3j, '0011': 1 - 3j, '0010': 3 - 3j},
                       "scatter": [-3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j,
                                   -3 + 1j, -1 + 1j, 1 + 1j, 3 + 1j,
                                   -3 - 1j, -1 - 1j, 1 - 1j, 3 - 1j,
                                   -3 - 3j, -1 - 3j, 1 - 3j, 3 - 3j]
                       }
        self.PSK_16 = {"dictionary": {'0000': np.exp(1j * np.pi * 0),      '0001': np.exp(1j * np.pi * 1 / 8),  '0011': np.exp(1j * np.pi * 2 / 8),  '0010': np.exp(1j * np.pi * 3 / 8),
                                      '0110': np.exp(1j * np.pi * 4 / 8),  '0111': np.exp(1j * np.pi * 5 / 8),  '0101': np.exp(1j * np.pi * 6 / 8),  '0100': np.exp(1j * np.pi * 7 / 8),
                                      '1100': np.exp(1j * np.pi * 8 / 8),  '1101': np.exp(1j * np.pi * 9 / 8),  '1111': np.exp(1j * np.pi * 10 / 8), '1110': np.exp(1j * np.pi * 11 / 8),
                                      '1010': np.exp(1j * np.pi * 12 / 8), '1011': np.exp(1j * np.pi * 13 / 8), '1001': np.exp(1j * np.pi * 14 / 8), '1000': np.exp(1j * np.pi * 15 / 8), },
                       "scatter": [np.exp(1j * np.pi * 0),      np.exp(1j * np.pi * 1 / 8),  np.exp(1j * np.pi * 2 / 8),  np.exp(1j * np.pi * 3 / 8),
                                   np.exp(1j * np.pi * 4 / 8),  np.exp(1j * np.pi * 5 / 8),  np.exp(1j * np.pi * 6 / 8),  np.exp(1j * np.pi * 7 / 8),
                                   np.exp(1j * np.pi * 8 / 8),  np.exp(1j * np.pi * 9 / 8),  np.exp(1j * np.pi * 10 / 8), np.exp(1j * np.pi * 11 / 8),
                                   np.exp(1j * np.pi * 12 / 8), np.exp(1j * np.pi * 13 / 8), np.exp(1j * np.pi * 14 / 8), np.exp(1j * np.pi * 15 / 8)]
                       }

        # 初始化时，将当前调制模式设为传入参数 mode
        if mode == "BPSK" or mode == "2PSK"or mode == "2_PSK" or mode == "PSK_2":
            self.mode = self.BPSK
        elif mode == "QPSK_A":
            self.mode = self.QPSK_A
        elif mode == "QPSK_B" or mode == "QPSK"or mode == "4QAM"or mode == "QAM" or mode == "QAM_4"or mode == "4_QAM":
            self.mode = self.QPSK_B
        elif mode == "PSK_8" or mode == "8PSK" or mode == "8_PSK"or mode == "PSK8":
            self.mode = self.PSK_8
        elif mode == "PSK_16" or mode == "16_PSK" or mode == "PSK16" or mode == "16PSK":
            self.mode = self.PSK_16
        elif mode == "QAM_16" or mode == "16_QAM" or mode == "16QAM" or mode == "QAM16":
            self.mode = self.QAM_16
        else:
            raise IOError("Check input modulation mode(检查输入的调制模式是否正确)")

        # 根据选择的调制方式计算参数
        self.bit = int(np.log2(len(self.mode["scatter"])))  # b = log2(s) 每个符号的比特数

        # 根据传入参数 norm 计算放缩比例
        if self.norm == "power_normalization_to_1":
            self.scatter_power = self.signal_power(self.mode["scatter"])
        elif self.norm == "power_normalization_to_1/Ns":
            self.scatter_power = np.sqrt(self.signal_power(self.mode["scatter"]) * self.Ns)
        else:
            raise IOError("Check input normalization mode(检查输入的归一化模式是否正确)")

    @staticmethod
    def signal_power(signal_input, *arg):
        """
        Calculate the power of the input signal, default axis=0
        :param signal_input:
        :param arg:
                    arg[0]: axis
        :return: power
        """
        signal_T = np.array(signal_input)
        energy = np.abs(signal_T) ** 2
        if len(arg) < 1:
            power = np.mean(energy)
        else:
            power = np.mean(energy, axis=arg[0])
        return power

    @staticmethod
    def signal_power_tf(signal_input, *arg):
        """
        Calculate the power of the input signal, default axis=0
        :param signal_input:
        :param arg:
                    arg[0]: axis
        :return: power
        """
        energy = tf.pow(tf.abs(signal_input), 2)
        if len(arg) < 1:
            power = tf.reduce_sum(tf.reduce_mean(energy))
        else:
            power = tf.reduce_sum(tf.reduce_mean(energy, axis=arg[0]))
        return power

    @staticmethod
    def db2power(db):
        """
        dB value to power
        :param db: dB value
        :return: power
        """
        power = np.power(10, db / 10)
        return power

    @staticmethod
    def power2db(power):
        """
         power to dB value
        :param power: power value
        :return: db: dB value
        """
        db = 10 * np.log10(power)
        return db

    @staticmethod
    def binary2str(data):
        """
        binary sequence  [0, 1, ..., 0] --to--> str "01...0"
        :return: str of binary sequence
        """
        bits = len(data)
        data = np.reshape(data, [-1, 1])  # 确保数据的维度只有一维
        data_str = str(data[0][0])
        for i in range(1, bits, 1):
            data_str = data_str + str(data[i][0])
        return data_str

    @staticmethod
    def count_accuracy_rate(binary_1, binary_2):
        """
        calculation the accuracy rate (bit error rate)
        :param binary_1:
        :param binary_2:
        :return: accuracy rate
        """
        binary_1 = np.array(binary_1)
        binary_2 = np.array(binary_2)
        error_num = np.sum(np.abs(binary_1 - binary_2))
        T = np.size(binary_1)
        bit_error_rate = error_num / int(np.size(binary_1))
        accuracy = 1. - bit_error_rate
        return accuracy

    @staticmethod
    def get_key(dictionary, value):
        """
        从字典dictionary中找到value对应的key值
        :param dictionary: 目的字典
        :param value: 查找的value
        :return: value对应的key
        """
        return [k for k, v in dictionary.items() if v == value]

    def noise_generator(self, power=None, SNR=None, size=None):
        """generate db dB noise with shape = shape , default signal power = 1
        """
        alpha = self.db2power(SNR)
        sigma = np.sqrt(power / alpha)  # 计算噪声标准差
        # 产生噪声
        noise_data = np.sqrt(0.5) * (np.random.normal(0, sigma, size=size) + np.random.normal(0, sigma, size=size) * 1j)
        noise_data = noise_data.astype(np.complex64)
        return noise_data

    def signal_generator(self, size=None):
        """
        "产生调制模式为 mode , 形状为 size 的复信号"
        :param size: size of output
        :return: binary_data, complex_signal_reshape, complex_signal_std
        """
        n = reduce(lambda x, y: x*y, size)  # 复信号信号长度
        binary_data = np.random.randint(2, size=(n, self.bit))  # 产生二进制序列 shape(n, b)
        binary_data_reshape = np.reshape(binary_data, [1, -1], order="C")
        complex_signal_std = []
        complex_signal_normalization = []
        for i in range(n):
            data_str = self.binary2str(binary_data[i])
            complex_signal = self.mode["dictionary"].get(data_str)
            complex_signal_std.append(complex_signal)
            complex_signal_normalization.append(complex_signal / self.scatter_power)
        complex_signal_reshape = np.reshape(complex_signal_normalization, size).astype(np.complex64)
        complex_signal_std = np.array(complex_signal_std).astype(np.complex64)
        scatter_std = self.mode["scatter"] / self.scatter_power
        return binary_data_reshape, complex_signal_reshape, complex_signal_std, scatter_std

    def signal_decoder(self, signal_receive):
        """
        dMap the input signal to the scatter diagram and demodulate the binary data --to--> [binary_data, complex_data]
        :param signal_receive: receive data
        :return: binary_data, complex_data_norm, complex_data_std
        """
        size = np.shape(signal_receive)
        signal_receive1 = np.array(signal_receive).reshape([-1, 1])
        # 进行星座映射前，将信号放缩回标准的幅度、
        signal_norm = signal_receive1 * self.scatter_power
        n = len(signal_receive1)  # 接收复信号信号长度
        binary_data = []
        complex_data_std = []
        for i in range(n):
            data_temp = signal_norm[i, :]
            distance = np.abs(data_temp - self.mode["scatter"])
            distance_min = np.min(distance)
            where = np.where(distance == distance_min)[0]
            index = int(where[0])
            result = self.mode["scatter"][index]
            complex_data_std.append(result)
            code = self.get_key(self.mode["dictionary"], result)
            for j in range(self.bit):
                binary_data.append(int(code[0][j]))
        complex_data = complex_data_std / self.scatter_power
        binary_data = np.reshape(binary_data, [1, -1])
        complex_data = np.reshape(complex_data, size).astype(np.complex64)
        complex_data_std = np.array(complex_data_std).astype(np.complex64)
        return binary_data, complex_data, complex_data_std
        pass



