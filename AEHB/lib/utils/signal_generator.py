import numpy as np


def x_pilot_and_noise_pilot(N, Ns, Nr, SNR, modulation_mode):
    """
    output_matrix = input_matrix * transmit_matrix
    :param input_matrix: N
    :param output_matrix: Nr
    :return: x_pilot
             noise_pilot
    """
    # x_pilot
    [_, _, _, signal] = signal_generator(N, Ns, modulation_mode, "power_normalization")
    power_signal = signal_power(signal)
    # alpha = db2power(SNR)
    # sigma = np.sqrt(power_signal / alpha)  # 计算噪声标准差
    # 产生噪声
    # noise_data = np.sqrt(0.5) * (np.random.normal(0, sigma, size=(N, Nr)) + np.random.normal(0, sigma, size=(N, Nr)) * 1j)
    noise_data = np.zeros([N, Nr])
    noise_data = noise_data.astype(np.complex64)
    return [signal, noise_data]

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
    bit_error_rate = error_num / int(len(binary_1))
    accuracy = 1. - bit_error_rate
    return accuracy

def linear_interpolation(h1, h2, n_inter):
    """
    Linear interpolation between h1 and h2
    :param h1: Channel matrix_1
    :param h2: Channel matrix_2
    :param n_inter: Interpolation number n_inter = n + 2
    :return: [h_1=h1, h_2, ..., h_n_inter=h2] array
    """
    h1 = np.array(h1)
    h2 = np.array(h2)
    n = n_inter - 1
    delta_h = (h2 - h1) / n
    h_sequence = [h1]
    # 初始化
    h = h1
    for i in range(1, n_inter):
        h = h + delta_h
        h_sequence.append(h)
    h_sequence = np.array(h_sequence)
    return h_sequence

def signal_power(signal):
    """
    Calculate the power of the input signal
    :param signal: input signal
    :return:power
    """
    signal = np.array(signal)
    power = np.mean((np.abs(signal) ** 2))
    return power

def db2power(db):
    """
    dB value to power
    :param db: dB value
    :return:
    """
    power = np.power(10, db / 10)
    return power

def signal_generator(frame_number, n_t, name=None, normalization=None):
    """
    binary signal turn to complex signal generation
    :param frame_number: Frame number
    :param n_t: Number of transmitting antennas
    :param name: Modulation mode
    :param normalization: Normalization method
    :return: Binary_data: binary sequence
              Complex_signal_std : complex_signal without normalization
              Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
    """
    n_f = frame_number * n_t

    def bpsk(n, normalize):
        """
        bpsk mode
        :param n: length of sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n))
        dictionary = {'1': 1 + 0j, '0': -1 + 0j}
        scatter = [1, -1]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [1, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i])
            complex_signal_std.append(dictionary.get(data_str))
            binary_data_reshape.append(binary_data[0][i])
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter / np.sqrt(scatter_power))
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    def qpsk_B(n, normalize):
        """
        qpsk_B mode
        :param n: length of sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n*2))
        dictionary = {'11': 1 + 1j, '10': 1 - 1j, '01': -1 + 1j, '00': -1 - 1j}
        scatter = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [2, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i]) + str(binary_data[1][i])
            # binary_data_reshape = [Num1_real, Num1_imag, Num2_real, Num2_imag, ... ,]
            binary_data_reshape.append(binary_data[0][i])
            binary_data_reshape.append(binary_data[1][i])

            complex_signal_std.append(dictionary.get(data_str))
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter / np.sqrt(scatter_power))
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    def qpsk_A(n, normalize):
        """
        qpsk_A mode
        :param n: length of sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n*2))
        dictionary = {'11': 1 + 0j, '10': 0 + 1j, '01': -1 + 0j, '00': 0 - 1j}
        scatter = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [2, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i]) + str(binary_data[1][i])
            # binary_data_reshape = [Num1_real, Num1_imag, Num2_real, Num2_imag, ... ,]
            binary_data_reshape.append(binary_data[0][i])
            binary_data_reshape.append(binary_data[1][i])

            complex_signal_std.append(dictionary.get(data_str))
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter / np.sqrt(scatter_power))
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    def qam_16(n, normalize):
        """
        qam_16 mode
        :param n: Binary sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n * 4))
        dictionary = {'1000': -3 + 3j, '1001': -1 + 3j, '1011': 1 + 3j, '1010': 3 + 3j,
                       '1100': -3 + 1j, '1101': -1 + 1j, '1111': 1 + 1j, '1110': 3 + 1j,
                       '0100': -3 - 1j, '0101': -1 - 1j, '0111': 1 - 1j, '0110': 3 - 1j,
                       '0000': -3 - 3j, '0001': -1 - 3j, '0011': 1 - 3j, '0010': 3 - 3j}
        scatter = [-3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j,
                   -3 + 1j, -1 + 1j, 1 + 1j, 3 + 1j,
                   -3 - 1j, -1 - 1j, 1 - 1j, 3 - 1j,
                   -3 - 3j, -1 - 3j, 1 - 3j, 3 - 3j]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [4, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i]) + str(binary_data[1][i]) + str(binary_data[2][i]) + str(binary_data[3][i])
            complex_signal_std.append(dictionary.get(data_str))
            binary_data_reshape.append(binary_data[0][i])
            binary_data_reshape.append(binary_data[1][i])
            binary_data_reshape.append(binary_data[2][i])
            binary_data_reshape.append(binary_data[3][i])
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter / np.sqrt(scatter_power))
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    def psk_8(n, normalize):
        """
        psk_8 mode
        :param n: Binary sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n * 3))
        dictionary = {'000': np.exp(1j * np.pi * 0),     '001': np.exp(1j * np.pi * 1 / 4), '011': np.exp(1j * np.pi * 2 / 4), '010': np.exp(1j * np.pi * 3 / 4),
                      '110': np.exp(1j * np.pi * 4 / 4), '111': np.exp(1j * np.pi * 5 / 4), '101': np.exp(1j * np.pi * 6 / 4), '100': np.exp(1j * np.pi * 7 / 4)}
        scatter = [np.exp(1j * np.pi * 0),     np.exp(1j * np.pi * 1 / 4), np.exp(1j * np.pi * 2 / 4), np.exp(1j * np.pi * 3 / 4),
                   np.exp(1j * np.pi * 4 / 4), np.exp(1j * np.pi * 5 / 4), np.exp(1j * np.pi * 6 / 4), np.exp(1j * np.pi * 7 / 4)]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [3, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i]) + str(binary_data[1][i]) + str(binary_data[2][i])
            complex_signal_std.append(dictionary.get(data_str))
            binary_data_reshape.append(binary_data[0][i])
            binary_data_reshape.append(binary_data[1][i])
            binary_data_reshape.append(binary_data[2][i])
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter) / np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    def psk_16(n, normalize):
        """
        psk_16 mode
        :param n: Binary sequence
        :param normalize: Normalization method
        :return: Binary_data: binary sequence
                  Complex_signal_std : complex_signal without normalization
                  Complex_signal_reshape: Complex sequence with shape[frame_number, n_t]
        """
        data = np.random.randint(2, size=(1, n * 4))
        dictionary = {'0000': np.exp(1j * np.pi * 0),      '0001': np.exp(1j * np.pi * 1 / 8),  '0011': np.exp(1j * np.pi * 2 / 8),  '0010': np.exp(1j * np.pi * 3 / 8),
                      '0110': np.exp(1j * np.pi * 4 / 8),  '0111': np.exp(1j * np.pi * 5 / 8),  '0101': np.exp(1j * np.pi * 6 / 8),  '0100': np.exp(1j * np.pi * 7 / 8),
                      '1100': np.exp(1j * np.pi * 8 / 8),  '1101': np.exp(1j * np.pi * 9 / 8),  '1111': np.exp(1j * np.pi * 10 / 8), '1110': np.exp(1j * np.pi * 11 / 8),
                      '1010': np.exp(1j * np.pi * 12 / 8), '1011': np.exp(1j * np.pi * 13 / 8), '1001': np.exp(1j * np.pi * 14 / 8), '1000': np.exp(1j * np.pi * 15 / 8), }
        scatter = [np.exp(1j * np.pi * 0),      np.exp(1j * np.pi * 1 / 8),  np.exp(1j * np.pi * 2 / 8),  np.exp(1j * np.pi * 3 / 8),
                   np.exp(1j * np.pi * 4 / 8),  np.exp(1j * np.pi * 5 / 8),  np.exp(1j * np.pi * 6 / 8),  np.exp(1j * np.pi * 7 / 8),
                   np.exp(1j * np.pi * 8 / 8),  np.exp(1j * np.pi * 9 / 8),  np.exp(1j * np.pi * 10 / 8), np.exp(1j * np.pi * 11 / 8),
                   np.exp(1j * np.pi * 12 / 8), np.exp(1j * np.pi * 13 / 8), np.exp(1j * np.pi * 14 / 8), np.exp(1j * np.pi * 15 / 8)]
        scatter_power = signal_power(scatter)
        binary_data = np.reshape(data, [4, -1])
        binary_data_reshape = []
        complex_signal_temp = []
        complex_signal_std = []
        for i in range(len(binary_data[0])):
            data_str = str(binary_data[0][i]) + str(binary_data[1][i]) + str(binary_data[2][i]) + str(binary_data[3][i])
            complex_signal_std.append(dictionary.get(data_str))
            binary_data_reshape.append(binary_data[0][i])
            binary_data_reshape.append(binary_data[1][i])
            binary_data_reshape.append(binary_data[2][i])
            binary_data_reshape.append(binary_data[3][i])
            if normalize == "power_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.sqrt(scatter_power))
                scatter_std = np.array(scatter / np.sqrt(scatter_power))
            elif normalize == "amplitude_normalization":
                complex_signal_temp.append(dictionary.get(data_str) / np.max(np.abs(scatter)))
                scatter_std = np.array(scatter / np.max(np.abs(scatter)))
            else:
                print("signal without normalization!\n")
                complex_signal_temp.append(dictionary.get(data_str))
                scatter_std = scatter
        complex_signal = np.array(complex_signal_temp)
        complex_signal_reshape = complex_signal.reshape(frame_number, n_t).astype(np.complex64)
        return [binary_data_reshape, scatter_std, complex_signal_std, complex_signal_reshape]

    # choosing modulation mode
    if name == "bpsk":
        return bpsk(n_f, normalization)
    elif name == "qpsk_A":
        return qpsk_A(n_f, normalization)
    elif name == "qpsk_B":
        return qpsk_B(n_f, normalization)
    elif name == "qam_16":
        return qam_16(n_f, normalization)
    elif name == "psk_8":
        return psk_8(n_f, normalization)
    elif name == "psk_16":
        return psk_16(n_f, normalization)
    else:
        print("Choosing right modulation mode:\n", "bpsk\n", "qpsk_A\n", "qpsk_B\n","qam_16\n", "psk_8\n", "psk_16\n")

def signal_decoder(data, name=None, normalization=None):
    """
    Map the input signal to the scatter diagram and demodulate the binary data
    :param data: input signal (complex)
    :param name: Modulation mode
    :param normalization: Normalization method
    :return: [binary_data, signal_std]
    """

    def get_key(dictionary, value):
        """
        从字典dictionary中找到value对应的key值
        :param dictionary: 目的字典
        :param value: 查找的value
        :return: value对应的key
        """
        return [k for k, v in dictionary.items() if v == value]

    def bpsk(signal, normalize=None):
        """
        bpsk mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'1': 1 + 0j, '0': -1 + 0j}
        scatter = [1 + 0j, -1 + 0j]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        T = len(signal[0])
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[int(np.where(distance == distance_min)[0])]
            complex_signal_temp.append(result)
            T = get_key(dictionary, result)[0][0]
            binary_data.append(int(T))
        return [binary_data, complex_signal_temp]

    def qpsk_B(signal, normalize=None):
        """
        bpsk_B mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'11': 1 + 1j, '10': 1 - 1j, '01': -1 + 1j, '00': -1 - 1j}
        scatter = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[int(np.where(distance == distance_min)[0])]
            complex_signal_temp.append(result)
            for j in range(2):
                T = get_key(dictionary, result)[0][j]
                binary_data.append(int(T))
        return [binary_data, complex_signal_temp]

    def qpsk_A(signal, normalize=None):
        """
        bpsk_A mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'11': 1 + 0j, '10': 0 + 1j, '01': -1 + 0j, '00': 0 - 1j}
        scatter = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[(np.where(distance == distance_min)[0])[0]]
            complex_signal_temp.append(result)
            for j in range(2):
                T = get_key(dictionary, result)[0][j]
                binary_data.append(int(T))
        return [binary_data, complex_signal_temp]

    def qam_16(signal, normalize=None):
        """
        bpsk mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'1000': -3 + 3j, '1001': -1 + 3j, '1011': 1 + 3j, '1010': 3 + 3j,
                       '1100': -3 + 1j, '1101': -1 + 1j, '1111': 1 + 1j, '1110': 3 + 1j,
                       '0100': -3 - 1j, '0101': -1 - 1j, '0111': 1 - 1j, '0110': 3 - 1j,
                       '0000': -3 - 3j, '0001': -1 - 3j, '0011': 1 - 3j, '0010': 3 - 3j}
        scatter = [-3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j,
                   - 3 + 1j, -1 + 1j, 1 + 1j, 3 + 1j,
                   - 3 - 1j, -1 - 1j, 1 - 1j, 3 - 1j,
                   - 3 - 3j, -1 - 3j, 1 - 3j, 3 - 3j]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[int(np.where(distance == distance_min)[0])]
            complex_signal_temp.append(result)
            for j in range(4):
                T = get_key(dictionary, result)[0][j]
                binary_data.append(int(T))
        return [binary_data, complex_signal_temp]

    def psk_8(signal, normalize=None):
        """
        psk_8 mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'000': np.exp(1j * np.pi * 0),   '001': np.exp(1j * np.pi * 1/4), '011': np.exp(1j * np.pi * 2/4), '010': np.exp(1j * np.pi * 3/4),
                      '110': np.exp(1j * np.pi * 4/4), '111': np.exp(1j * np.pi * 5/4), '101': np.exp(1j * np.pi * 6/4), '100': np.exp(1j * np.pi * 7/4)}
        scatter = [np.exp(1j * np.pi * 0),   np.exp(1j * np.pi * 1/4), np.exp(1j * np.pi * 2/4), np.exp(1j * np.pi * 3/4),
                   np.exp(1j * np.pi * 4/4), np.exp(1j * np.pi * 5/4), np.exp(1j * np.pi * 6/4), np.exp(1j * np.pi * 7/4)]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * 2*np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[(np.where(distance == distance_min)[0])[0]]
            complex_signal_temp.append(result)
            for j in range(3):
                T = get_key(dictionary, result)[0][j]
                binary_data.append(int(T))
        return [binary_data, complex_signal_temp]

    def psk_16(signal, normalize=None):
        """
        psk_16 mode
        :param signal: input complex signal
        :param normalize: Normalization method
        :return: [binary_data, signal_std]
        """
        signal = np.array(signal).reshape([1, -1])
        dictionary = {'0000': np.exp(1j * np.pi * 0),      '0001': np.exp(1j * np.pi * 1 / 8),  '0011': np.exp(1j * np.pi * 2 / 8),  '0010': np.exp(1j * np.pi * 3 / 8),
                      '0110': np.exp(1j * np.pi * 4 / 8),  '0111': np.exp(1j * np.pi * 5 / 8),  '0101': np.exp(1j * np.pi * 6 / 8),  '0100': np.exp(1j * np.pi * 7 / 8),
                      '1100': np.exp(1j * np.pi * 8 / 8),  '1101': np.exp(1j * np.pi * 9 / 8),  '1111': np.exp(1j * np.pi * 10 / 8), '1110': np.exp(1j * np.pi * 11 / 8),
                      '1010': np.exp(1j * np.pi * 12 / 8), '1011': np.exp(1j * np.pi * 13 / 8), '1001': np.exp(1j * np.pi * 14 / 8), '1000': np.exp(1j * np.pi * 15 / 8), }
        scatter = [np.exp(1j * np.pi * 0),      np.exp(1j * np.pi * 1 / 8),  np.exp(1j * np.pi * 2 / 8),  np.exp(1j * np.pi * 3 / 8),
                   np.exp(1j * np.pi * 4 / 8),  np.exp(1j * np.pi * 5 / 8),  np.exp(1j * np.pi * 6 / 8),  np.exp(1j * np.pi * 7 / 8),
                   np.exp(1j * np.pi * 8 / 8),  np.exp(1j * np.pi * 9 / 8),  np.exp(1j * np.pi * 10 / 8), np.exp(1j * np.pi * 11 / 8),
                   np.exp(1j * np.pi * 12 / 8), np.exp(1j * np.pi * 13 / 8), np.exp(1j * np.pi * 14 / 8), np.exp(1j * np.pi * 15 / 8)]
        scatter_power = signal_power(scatter)
        complex_signal_temp = []
        binary_data = []
        for i in range(len(signal[0])):
            if normalize == "power_normalization":
                data_temp = signal[0, i] * np.sqrt(scatter_power)
            elif normalize == "amplitude_normalization":
                data_temp = signal[0, i] * np.max(np.abs(scatter))
            else:
                print("signal without normalization!\n")
                data_temp = signal[0, i]
            distance = np.abs(data_temp - scatter)
            distance_min = np.min(distance)
            result = scatter[(np.where(distance == distance_min)[0])[0]]
            complex_signal_temp.append(result)
            for j in range(4):
                T = get_key(dictionary, result)[0][j]
                binary_data.append(int(T))
        return [binary_data, complex_signal_temp]
    # choosing modulation mode
    if name == "bpsk":
        return bpsk(data, normalization)
    elif name == "qpsk_A":
        return qpsk_A(data, normalization)
    elif name == "qpsk_B":
        return qpsk_B(data, normalization)
    elif name == "qam_16":
        return qam_16(data, normalization)
    elif name == "psk_8":
        return psk_8(data, normalization)
    elif name == "psk_16":
        return psk_16(data, normalization)
    else:
        print("Choosing right modulation mode:\n", "bpsk\n", "qpsk\n", "qam_16\n")


# # test
# [binary_input, scatter_std, complex_std_input, complex_data] = signal_generator(4, 4, "psk_8", "power_normalization")
# print("binary_input:\n", binary_input)
# print("scatter_std:\n", scatter_std)
# print("complex_std_input:\n", complex_std_input)
# print("complex_data:\n", complex_data)
# [binary_output, complex_output] = signal_decoder(complex_data, "psk_8", "power_normalization")
# print("binary_output:\n", binary_output)
# print("complex_output:\n", complex_output)
# pass




