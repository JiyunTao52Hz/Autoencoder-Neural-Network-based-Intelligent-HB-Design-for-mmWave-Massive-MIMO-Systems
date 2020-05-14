import tensorflow as tf


def phase_shift_matmul(input_signal, input_phase, name=None):
    """
    analog phase shift， 矩阵乘法
    :param input_signal: baseband signal
    :param input_phase: theta
    :param name: name
    :return: RF signal = baseband signal * (cos(theta) + 1j*sin(theta))
    """
    return tf.matmul(input_signal, tf.complex(tf.cos(input_phase), tf.sin(input_phase)), name=name)


def phase_shift_multiply(input_signal, input_phase, name=None):
    """
    analog phase shift, 对应位置相乘
    :param input_signal: baseband signal
    :param input_phase: theta
    :param name: name
    :return: RF signal = baseband signal * (cos(theta) + 1j*sin(theta))
    """
    return tf.multiply(input_signal, tf.complex(tf.cos(input_phase), tf.sin(input_phase)), name=name)


def power_constrained(input_signal, constrained=None):
    """
    对输入的信号进行功率约束，约束到 power 大小 ouput_signal = power*(input / signal_power(input_signal))
    :param input_signal:
    :param constrained:
    :param power:
    :return:
    """
    if constrained:
        signal_power = tf.reduce_sum(tf.reduce_mean(tf.square(tf.abs(input_signal)), axis=0))
        parameter = tf.sqrt(signal_power)
        T = tf.real(input_signal)
        T1 = tf.imag(input_signal)
        signal_real = tf.div(tf.real(input_signal), parameter)
        signal_imag = tf.div(tf.imag(input_signal), parameter)
        output_signal = tf.complex(signal_real, signal_imag)
    else:
        output_signal = input_signal

    return output_signal