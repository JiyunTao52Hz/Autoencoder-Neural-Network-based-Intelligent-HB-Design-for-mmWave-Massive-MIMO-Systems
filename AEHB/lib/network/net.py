import tensorflow as tf
import tensorflow.contrib.slim as slim
import lib.utils.beamforming_tools as bt
import lib.utils.communication_tools as ct
import lib.config.config as cfg
import numpy as np

signal = ct.signal(cfg.FLAGS.modulation_mode, cfg.FLAGS.power_normalization, cfg.FLAGS.Ns)


class CdnnWithBN1Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_L1 = tf.layers.batch_normalization(real_L1_f, training=self.training)
                # real_baseband_output
                real_baseband_output = real_L1
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=self.training)
                # imag_baseband_output
                imag_baseband_output = imag_L1
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=None, scope='real_decoding__L1')
                real_prediction = real_decoding_L1_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=None, scope='imag_decoding__L1')
                imag_prediction = imag_decoding_L1_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithoutBN1Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_baseband_output = real_L1_f
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_baseband_output = imag_L1_f
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=None, scope='real_decoding__L1')
                real_prediction = real_decoding_L1_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=None, scope='imag_decoding__L1')
                imag_prediction = imag_decoding_L1_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithBN2Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_L1 = tf.layers.batch_normalization(real_L1_f, training=self.training)
                # real_L2
                real_L2_f = slim.fully_connected(real_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                real_L2 = tf.layers.batch_normalization(real_L2_f, training=self.training)
                # real_baseband_output
                real_baseband_output = real_L2
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=self.training)
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                imag_L2 = tf.layers.batch_normalization(imag_L2_f, training=self.training)
                # imag_baseband_output
                imag_baseband_output = imag_L2
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                real_decoding_L1 = tf.layers.batch_normalization(real_decoding_L1_f, training=self.training)
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1, self.baseband_decoding_L2, activation_fn=None, scope='real_decoding__L2')
                real_prediction = real_decoding_L2_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                imag_decoding_L1 = tf.layers.batch_normalization(imag_decoding_L1_f, training=self.training)
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1, self.baseband_decoding_L2, activation_fn=None, scope='imag_decoding__L2')
                imag_prediction = imag_decoding_L2_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithoutBN2Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                # real_L2
                real_L2_f = slim.fully_connected(real_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                real_baseband_output = real_L2_f
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                imag_baseband_output = imag_L2_f
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1_f, self.baseband_decoding_L2, activation_fn=None, scope='real_decoding__L2')
                real_prediction = real_decoding_L2_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1_f, self.baseband_decoding_L2, activation_fn=None, scope='imag_decoding__L2')
                imag_prediction = imag_decoding_L2_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithBN3Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 2 * self.N
        self.baseband_precoding_L3 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 2 * self.N
        self.baseband_decoding_L3 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_L1 = tf.layers.batch_normalization(real_L1_f, training=self.training)
                # real_L2
                real_L2_f = slim.fully_connected(real_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                real_L2 = tf.layers.batch_normalization(real_L2_f, training=self.training)
                # real_L3
                real_L3_f = slim.fully_connected(real_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                real_L3 = tf.layers.batch_normalization(real_L3_f, training=self.training)
                # real_baseband_output
                real_baseband_output = real_L3
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=self.training)
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                imag_L2 = tf.layers.batch_normalization(imag_L2_f, training=self.training)
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                imag_L3 = tf.layers.batch_normalization(imag_L3_f, training=self.training)
                # imag_baseband_output
                imag_baseband_output = imag_L3
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                real_decoding_L1 = tf.layers.batch_normalization(real_decoding_L1_f, training=self.training)
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding__L2')
                real_decoding_L2 = tf.layers.batch_normalization(real_decoding_L2_f, training=self.training)
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2, self.baseband_decoding_L3, activation_fn=None, scope='real_decoding_L3')
                real_prediction = real_decoding_L3_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                imag_decoding_L1 = tf.layers.batch_normalization(imag_decoding_L1_f, training=self.training)
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding__L2')
                imag_decoding_L2 = tf.layers.batch_normalization(imag_decoding_L2_f, training=self.training)
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2, self.baseband_decoding_L3, activation_fn=None, scope='imag_decoding_L3')
                imag_prediction = imag_decoding_L3_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithoutBN3Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 2 * self.N
        self.baseband_precoding_L3 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 2 * self.N
        self.baseband_decoding_L3 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                # real_L2
                real_L2_f = slim.fully_connected(real_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                # real_L3
                real_L3_f = slim.fully_connected(real_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                # real_baseband_output
                real_baseband_output = real_L3_f
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                # imag_baseband_output
                imag_baseband_output = imag_L3_f
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding__L2')
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2_f, self.baseband_decoding_L3, activation_fn=None, scope='real_decoding__L3')
                real_prediction = real_decoding_L3_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding__L2')
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2_f, self.baseband_decoding_L3, activation_fn=None, scope='imag_decoding__L3')
                imag_prediction = imag_decoding_L3_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithBN5Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 4 * self.N
        self.baseband_precoding_L3 = self.Ns * 4 * self.N
        self.baseband_precoding_L4 = self.Ns * 2 * self.N
        self.baseband_precoding_L5 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 4 * self.N
        self.baseband_decoding_L3 = self.Ns * 4 * self.N
        self.baseband_decoding_L4 = self.Ns * 2 * self.N
        self.baseband_decoding_L5 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_L1 = tf.layers.batch_normalization(real_L1_f, training=self.training)
                # real_L2
                real_L2_f = slim.fully_connected(real_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                real_L2 = tf.layers.batch_normalization(real_L2_f, training=self.training)
                # real_L3
                real_L3_f = slim.fully_connected(real_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                real_L3 = tf.layers.batch_normalization(real_L3_f, training=self.training)
                # real_L4
                real_L4_f = slim.fully_connected(real_L3, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='real_L4')
                real_L4 = tf.layers.batch_normalization(real_L4_f, training=self.training)
                # real_L5
                real_L5_f = slim.fully_connected(real_L4, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='real_L5')
                real_L5 = tf.layers.batch_normalization(real_L5_f, training=self.training)
                # real_baseband_output
                real_baseband_output = real_L5
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=self.training)
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                imag_L2 = tf.layers.batch_normalization(imag_L2_f, training=self.training)
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                imag_L3 = tf.layers.batch_normalization(imag_L3_f, training=self.training)
                # imag_L4
                imag_L4_f = slim.fully_connected(imag_L3, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='imag_L4')
                imag_L4 = tf.layers.batch_normalization(imag_L4_f, training=self.training)
                # imag_L5
                imag_L5_f = slim.fully_connected(imag_L4, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='imag_L5')
                imag_L5 = tf.layers.batch_normalization(imag_L5_f, training=self.training)
                # imag_baseband_output
                imag_baseband_output = imag_L5
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                real_decoding_L1 = tf.layers.batch_normalization(real_decoding_L1_f, training=self.training)
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding__L2')
                real_decoding_L2 = tf.layers.batch_normalization(real_decoding_L2_f, training=self.training)
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='real_decoding__L3')
                real_decoding_L3 = tf.layers.batch_normalization(real_decoding_L3_f, training=self.training)
                # real_L4
                real_decoding_L4_f = slim.fully_connected(real_decoding_L3, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='real_decoding__L4')
                real_decoding_L4 = tf.layers.batch_normalization(real_decoding_L4_f, training=self.training)
                # real_L5
                real_decoding_L5_f = slim.fully_connected(real_decoding_L4, self.baseband_decoding_L5, activation_fn=None, scope='real_decoding__L5')
                real_prediction = real_decoding_L5_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                imag_decoding_L1 = tf.layers.batch_normalization(imag_decoding_L1_f, training=self.training)
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding__L2')
                imag_decoding_L2 = tf.layers.batch_normalization(imag_decoding_L2_f, training=self.training)
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='imag_decoding__L3')
                imag_decoding_L3 = tf.layers.batch_normalization(imag_decoding_L3_f, training=self.training)
                # imag_L4
                imag_decoding_L4_f = slim.fully_connected(imag_decoding_L3, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='imag_decoding__L4')
                imag_decoding_L4 = tf.layers.batch_normalization(imag_decoding_L4_f, training=self.training)
                # imag_L5
                imag_decoding_L5_f = slim.fully_connected(imag_decoding_L4, self.baseband_decoding_L5, activation_fn=None, scope='imag_decoding__L5')
                imag_prediction = imag_decoding_L5_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithoutBN5Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 4 * self.N
        self.baseband_precoding_L3 = self.Ns * 4 * self.N
        self.baseband_precoding_L4 = self.Ns * 2 * self.N
        self.baseband_precoding_L5 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 4 * self.N
        self.baseband_decoding_L3 = self.Ns * 4 * self.N
        self.baseband_decoding_L4 = self.Ns * 2 * self.N
        self.baseband_decoding_L5 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                # real_L2
                real_L2_f = slim.fully_connected(real_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                # real_L3
                real_L3_f = slim.fully_connected(real_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                # real_L4
                real_L4_f = slim.fully_connected(real_L3_f, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='real_L4')
                # real_L5
                real_L5_f = slim.fully_connected(real_L4_f, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='real_L5')
                # real_baseband_output
                real_baseband_output = real_L5_f
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                # imag_L4
                imag_L4_f = slim.fully_connected(imag_L3_f, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='imag_L4')
                # imag_L5
                imag_L5_f = slim.fully_connected(imag_L4_f, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='imag_L5')
                # imag_baseband_output
                imag_baseband_output = imag_L5_f
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding__L1')
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding__L2')
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2_f, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='real_decoding__L3')
                # real_L4
                real_decoding_L4_f = slim.fully_connected(real_decoding_L3_f, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='real_decoding__L4')
                # real_L5
                real_decoding_L5_f = slim.fully_connected(real_decoding_L4_f, self.baseband_decoding_L5, activation_fn=None, scope='real_decoding__L5')
                real_prediction = real_decoding_L5_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding__L1')
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding__L2')
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2_f, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='imag_decoding__L3')
                # imag_L4
                imag_decoding_L4_f = slim.fully_connected(imag_decoding_L3_f, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='imag_decoding__L4')
                # imag_L5
                imag_decoding_L5_f = slim.fully_connected(imag_decoding_L4_f, self.baseband_decoding_L5, activation_fn=None, scope='imag_decoding__L5')
                imag_prediction = imag_decoding_L5_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithBN9Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 4 * self.N
        self.baseband_precoding_L3 = self.Ns * 8 * self.N
        self.baseband_precoding_L4 = self.Ns * 16 * self.N
        self.baseband_precoding_L5 = self.Ns * 16 * self.N
        self.baseband_precoding_L6 = self.Ns * 8 * self.N
        self.baseband_precoding_L7 = self.Ns * 4 * self.N
        self.baseband_precoding_L8 = self.Ns * 2 * self.N
        self.baseband_precoding_L9 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 4 * self.N
        self.baseband_decoding_L3 = self.Ns * 8 * self.N
        self.baseband_decoding_L4 = self.Ns * 16 * self.N
        self.baseband_decoding_L5 = self.Ns * 16 * self.N
        self.baseband_decoding_L6 = self.Ns * 8 * self.N
        self.baseband_decoding_L7 = self.Ns * 4 * self.N
        self.baseband_decoding_L8 = self.Ns * 2 * self.N
        self.baseband_decoding_L9 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                real_L1 = tf.layers.batch_normalization(real_L1_f, training=self.training)
                # real_L2
                real_L2_f = slim.fully_connected(real_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                real_L2 = tf.layers.batch_normalization(real_L2_f, training=self.training)
                # real_L3
                real_L3_f = slim.fully_connected(real_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                real_L3 = tf.layers.batch_normalization(real_L3_f, training=self.training)
                # real_L4
                real_L4_f = slim.fully_connected(real_L3, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='real_L4')
                real_L4 = tf.layers.batch_normalization(real_L4_f, training=self.training)
                # real_L5
                real_L5_f = slim.fully_connected(real_L4, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='real_L5')
                real_L5 = tf.layers.batch_normalization(real_L5_f, training=self.training)
                # real_L6
                real_L6_f = slim.fully_connected(real_L5, self.baseband_precoding_L6, activation_fn=self.activation_function, scope='real_L6')
                real_L6 = tf.layers.batch_normalization(real_L6_f, training=self.training)
                # real_L7
                real_L7_f = slim.fully_connected(real_L6, self.baseband_precoding_L7, activation_fn=self.activation_function, scope='real_L7')
                real_L7 = tf.layers.batch_normalization(real_L7_f, training=self.training)
                # real_L8
                real_L8_f = slim.fully_connected(real_L7, self.baseband_precoding_L8, activation_fn=self.activation_function, scope='real_L8')
                real_L8 = tf.layers.batch_normalization(real_L8_f, training=self.training)
                # real_L9
                real_L9_f = slim.fully_connected(real_L8, self.baseband_precoding_L9, activation_fn=self.activation_function, scope='real_L9')
                real_L9 = tf.layers.batch_normalization(real_L9_f, training=self.training)
                # real_baseband_output
                real_baseband_output = real_L9
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=self.training)
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                imag_L2 = tf.layers.batch_normalization(imag_L2_f, training=self.training)
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                imag_L3 = tf.layers.batch_normalization(imag_L3_f, training=self.training)
                # imag_L4
                imag_L4_f = slim.fully_connected(imag_L3, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='imag_L4')
                imag_L4 = tf.layers.batch_normalization(imag_L4_f, training=self.training)
                # imag_L5
                imag_L5_f = slim.fully_connected(imag_L4, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='imag_L5')
                imag_L5 = tf.layers.batch_normalization(imag_L5_f, training=self.training)
                # imag_L6
                imag_L6_f = slim.fully_connected(imag_L5, self.baseband_precoding_L6, activation_fn=self.activation_function, scope='imag_L6')
                imag_L6 = tf.layers.batch_normalization(imag_L6_f, training=self.training)
                # imag_L7
                imag_L7_f = slim.fully_connected(imag_L6, self.baseband_precoding_L7, activation_fn=self.activation_function, scope='imag_L7')
                imag_L7 = tf.layers.batch_normalization(imag_L7_f, training=self.training)
                # imag_L8
                imag_L8_f = slim.fully_connected(imag_L7, self.baseband_precoding_L8, activation_fn=self.activation_function, scope='imag_L8')
                imag_L8 = tf.layers.batch_normalization(imag_L8_f, training=self.training)
                # imag_L9
                imag_L9_f = slim.fully_connected(imag_L8, self.baseband_precoding_L9, activation_fn=self.activation_function, scope='imag_L9')
                imag_L9 = tf.layers.batch_normalization(imag_L9_f, training=self.training)
                # imag_baseband_output
                imag_baseband_output = imag_L9
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding_L1')
                real_decoding_L1 = tf.layers.batch_normalization(real_decoding_L1_f, training=self.training)
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding_L2')
                real_decoding_L2 = tf.layers.batch_normalization(real_decoding_L2_f, training=self.training)
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='real_decoding_L3')
                real_decoding_L3 = tf.layers.batch_normalization(real_decoding_L3_f, training=self.training)
                # real_L4
                real_decoding_L4_f = slim.fully_connected(real_decoding_L3, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='real_decoding_L4')
                real_decoding_L4 = tf.layers.batch_normalization(real_decoding_L4_f, training=self.training)
                # real_L5
                real_decoding_L5_f = slim.fully_connected(real_decoding_L4, self.baseband_decoding_L5, activation_fn=self.activation_function, scope='real_decoding_L5')
                real_decoding_L5 = tf.layers.batch_normalization(real_decoding_L5_f, training=self.training)
                # real_L6
                real_decoding_L6_f = slim.fully_connected(real_decoding_L5, self.baseband_decoding_L6, activation_fn=self.activation_function, scope='real_decoding_L6')
                real_decoding_L6 = tf.layers.batch_normalization(real_decoding_L6_f, training=self.training)
                # real_L7
                real_decoding_L7_f = slim.fully_connected(real_decoding_L6, self.baseband_decoding_L7, activation_fn=self.activation_function, scope='real_decoding_L7')
                real_decoding_L7 = tf.layers.batch_normalization(real_decoding_L7_f, training=self.training)
                # real_L8
                real_decoding_L8_f = slim.fully_connected(real_decoding_L7, self.baseband_decoding_L8, activation_fn=self.activation_function, scope='real_decoding_L8')
                real_decoding_L8 = tf.layers.batch_normalization(real_decoding_L8_f, training=self.training)
                # real_L9
                real_decoding_L9_f = slim.fully_connected(real_decoding_L8, self.baseband_decoding_L9, activation_fn=None, scope='real_decoding_L9')
                real_prediction = real_decoding_L9_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding_L1')
                imag_decoding_L1 = tf.layers.batch_normalization(imag_decoding_L1_f, training=self.training)
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding_L2')
                imag_decoding_L2 = tf.layers.batch_normalization(imag_decoding_L2_f, training=self.training)
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='imag_decoding_L3')
                imag_decoding_L3 = tf.layers.batch_normalization(imag_decoding_L3_f, training=self.training)
                # imag_L4
                imag_decoding_L4_f = slim.fully_connected(imag_decoding_L3, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='imag_decoding_L4')
                imag_decoding_L4 = tf.layers.batch_normalization(imag_decoding_L4_f, training=self.training)
                # imag_L5
                imag_decoding_L5_f = slim.fully_connected(imag_decoding_L4, self.baseband_decoding_L5, activation_fn=self.activation_function, scope='imag_decoding_L5')
                imag_decoding_L5 = tf.layers.batch_normalization(imag_decoding_L5_f, training=self.training)
                # imag_L6
                imag_decoding_L6_f = slim.fully_connected(imag_decoding_L5, self.baseband_decoding_L6, activation_fn=self.activation_function, scope='imag_decoding_L6')
                imag_decoding_L6 = tf.layers.batch_normalization(imag_decoding_L6_f, training=self.training)
                # imag_L7
                imag_decoding_L7_f = slim.fully_connected(imag_decoding_L6, self.baseband_decoding_L7, activation_fn=self.activation_function, scope='imag_decoding_L7')
                imag_decoding_L7 = tf.layers.batch_normalization(imag_decoding_L7_f, training=self.training)
                # imag_L8
                imag_decoding_L8_f = slim.fully_connected(imag_decoding_L7, self.baseband_decoding_L8, activation_fn=self.activation_function, scope='imag_decoding_L8')
                imag_decoding_L8 = tf.layers.batch_normalization(imag_decoding_L8_f, training=self.training)
                # imag_L9
                imag_decoding_L9_f = slim.fully_connected(imag_decoding_L8, self.baseband_decoding_L9, activation_fn=None, scope='imag_decoding_L9')
                imag_prediction = imag_decoding_L9_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged


class CdnnWithoutBN9Layers:
    """
    constrained deep neural network, 8 RF chains , with batch normalization
    """
    def __init__(self, activation_function, N):
        """
        初始化网络参数（每层神经网络的神经元个数）
        """
        self.N = N
        self.Ns = cfg.FLAGS.Ns
        self.Nr = cfg.FLAGS.Nr
        self.Nt = cfg.FLAGS.Nt
        self.Ntrf = cfg.FLAGS.Ntrf
        self.Nrrf = cfg.FLAGS.Nrrf
        self.baseband_precoding_L0 = self.Ns
        self.baseband_precoding_L1 = self.Ns * 2 * self.N
        self.baseband_precoding_L2 = self.Ns * 4 * self.N
        self.baseband_precoding_L3 = self.Ns * 8 * self.N
        self.baseband_precoding_L4 = self.Ns * 16 * self.N
        self.baseband_precoding_L5 = self.Ns * 16 * self.N
        self.baseband_precoding_L6 = self.Ns * 8 * self.N
        self.baseband_precoding_L7 = self.Ns * 4 * self.N
        self.baseband_precoding_L8 = self.Ns * 2 * self.N
        self.baseband_precoding_L9 = self.Ntrf
        self.RF_precoding_L1 = self.Nt
        self.RF_decoding_L1 = self.Nr
        self.baseband_decoding_L0 = self.Nrrf
        self.baseband_decoding_L1 = self.Ns * 2 * self.N
        self.baseband_decoding_L2 = self.Ns * 4 * self.N
        self.baseband_decoding_L3 = self.Ns * 8 * self.N
        self.baseband_decoding_L4 = self.Ns * 16 * self.N
        self.baseband_decoding_L5 = self.Ns * 16 * self.N
        self.baseband_decoding_L6 = self.Ns * 8 * self.N
        self.baseband_decoding_L7 = self.Ns * 4 * self.N
        self.baseband_decoding_L8 = self.Ns * 2 * self.N
        self.baseband_decoding_L9 = self.Ns
        # 定义激活函数
        self.activation_function = activation_function
        # 定义一个字典，用于存储网络需要返回的参数
        self.predictions = {}

    def __call__(self, x, H, noise, training, learning_rate):
        """
        定义传入的参数
        :param x:
        :param H:
        :param noise:
        :param training:
        :return:
        """
        self.x = x
        self.H = H
        self.noise = noise
        self.training = training
        self.Learning_rate = learning_rate

        # build network
        with tf.name_scope('Baseband_Precoding'):
            with tf.name_scope('Real_channel'):
                # real_L1
                real_L1_f = slim.fully_connected(tf.real(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='real_L1')
                # real_L2
                real_L2_f = slim.fully_connected(real_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='real_L2')
                # real_L3
                real_L3_f = slim.fully_connected(real_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='real_L3')
                # real_L4
                real_L4_f = slim.fully_connected(real_L3_f, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='real_L4')
                # real_L5
                real_L5_f = slim.fully_connected(real_L4_f, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='real_L5')
                # real_L6
                real_L6_f = slim.fully_connected(real_L5_f, self.baseband_precoding_L6, activation_fn=self.activation_function, scope='real_L6')
                # real_L7
                real_L7_f = slim.fully_connected(real_L6_f, self.baseband_precoding_L7, activation_fn=self.activation_function, scope='real_L7')
                # real_L8
                real_L8_f = slim.fully_connected(real_L7_f, self.baseband_precoding_L8, activation_fn=self.activation_function, scope='real_L8')
                # real_L9
                real_L9_f = slim.fully_connected(real_L8_f, self.baseband_precoding_L9, activation_fn=self.activation_function, scope='real_L9')
                # real_baseband_output
                real_baseband_output = real_L9_f
            with tf.name_scope('Imag_channel'):
                # imag_L1
                imag_L1_f = slim.fully_connected(tf.imag(self.x), self.baseband_precoding_L1, activation_fn=self.activation_function, scope='imag_L1')
                # imag_L2
                imag_L2_f = slim.fully_connected(imag_L1_f, self.baseband_precoding_L2, activation_fn=self.activation_function, scope='imag_L2')
                # imag_L3
                imag_L3_f = slim.fully_connected(imag_L2_f, self.baseband_precoding_L3, activation_fn=self.activation_function, scope='imag_L3')
                # imag_L4
                imag_L4_f = slim.fully_connected(imag_L3_f, self.baseband_precoding_L4, activation_fn=self.activation_function, scope='imag_L4')
                # imag_L5
                imag_L5_f = slim.fully_connected(imag_L4_f, self.baseband_precoding_L5, activation_fn=self.activation_function, scope='imag_L5')
                # imag_L6
                imag_L6_f = slim.fully_connected(imag_L5_f, self.baseband_precoding_L6, activation_fn=self.activation_function, scope='imag_L6')
                # imag_L7
                imag_L7_f = slim.fully_connected(imag_L6_f, self.baseband_precoding_L7, activation_fn=self.activation_function, scope='imag_L7')
                # imag_L8
                imag_L8_f = slim.fully_connected(imag_L7_f, self.baseband_precoding_L8, activation_fn=self.activation_function, scope='imag_L8')
                # imag_L9
                imag_L9_f = slim.fully_connected(imag_L8_f, self.baseband_precoding_L9, activation_fn=self.activation_function, scope='imag_L9')
                # imag_baseband_output
                imag_baseband_output = imag_L9_f
            # baseband_precoding_output
            baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

        with tf.name_scope('RF_Precoding'):
            n_p1 = tf.to_int32(self.RF_precoding_L1 / self.Ntrf)  # 一条RF链路映射到 n_p1 个发射天线上
            theta_precoding_L1 = tf.Variable(tf.random_uniform([self.Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")

            with tf.name_scope('layer_1'):
                # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
                # L1_Floor1
                RF_precoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 0], (-1, 1)), tf.reshape(theta_precoding_L1[0, :], (1, -1)), name="RF_L1_F1")
                # L1_Floor2
                RF_precoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 1], (-1, 1)), tf.reshape(theta_precoding_L1[1, :], (1, -1)), name="RF_L1_F2")
                # L1_Floor3
                RF_precoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 2], (-1, 1)), tf.reshape(theta_precoding_L1[2, :], (1, -1)), name="RF_L1_F3")
                # L1_Floor4
                RF_precoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 3], (-1, 1)), tf.reshape(theta_precoding_L1[3, :], (1, -1)), name="RF_L1_F4")
                # L1_Floor5
                RF_precoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 4], (-1, 1)), tf.reshape(theta_precoding_L1[4, :], (1, -1)), name="RF_L1_F5")
                # L1_Floor6
                RF_precoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 5], (-1, 1)), tf.reshape(theta_precoding_L1[5, :], (1, -1)), name="RF_L1_F6")
                # L1_Floor7
                RF_precoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 6], (-1, 1)), tf.reshape(theta_precoding_L1[6, :], (1, -1)), name="RF_L1_F7")
                # L1_Floor8
                RF_precoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(baseband_precoding_output[:, 7], (-1, 1)), tf.reshape(theta_precoding_L1[7, :], (1, -1)), name="RF_L1_F8")
                #  将每个Floor的数据连在一起
            RF_precoding_output = tf.concat([RF_precoding_L1_F1, RF_precoding_L1_F2, RF_precoding_L1_F3, RF_precoding_L1_F4,
                                             RF_precoding_L1_F5, RF_precoding_L1_F6, RF_precoding_L1_F7, RF_precoding_L1_F8
                                             ], 1, name="RF_output")  # 两层

        with tf.name_scope('Power_Constrained'):
            power_constrained_output = bt.power_constrained(RF_precoding_output, cfg.FLAGS.constrained)

        with tf.name_scope('Channel_Transmission'):
            # 过传输矩阵H
            real_H_temp = tf.matmul(tf.real(power_constrained_output), tf.real(H), name="RxR") - tf.matmul(tf.imag(power_constrained_output), tf.imag(H), name="IxI")
            imag_H_temp = tf.matmul(tf.real(power_constrained_output), tf.imag(H), name="RxI") + tf.matmul(tf.imag(power_constrained_output), tf.real(H), name="IxR")
            H_output = tf.complex(real_H_temp, imag_H_temp, name="H_output")
        # add noise
        with tf.name_scope("add_noise"):
            real_noise_output = tf.add(real_H_temp, tf.real(noise), name="real")
            imag_noise_output = tf.add(imag_H_temp, tf.imag(noise), name="imag")
            # H_output
            noise_output = tf.complex(real_noise_output, imag_noise_output, name="H_output")

        with tf.name_scope('RF_decoding'):
            n_d1 = tf.to_int32(self.RF_decoding_L1 / self.Nrrf)
            theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, self.Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
            with tf.name_scope('Layer_1'):
                # L2_Floor1
                RF_decoding_L1_F1 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
                # L2_Floor2
                RF_decoding_L1_F2 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
                # L2_Floor3
                RF_decoding_L1_F3 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
                # L2_Floor4
                RF_decoding_L1_F4 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
                # L2_Floor5
                RF_decoding_L1_F5 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
                # L2_Floor6
                RF_decoding_L1_F6 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
                # L2_Floor7
                RF_decoding_L1_F7 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
                # L2_Floor8
                RF_decoding_L1_F8 = bt.phase_shift_matmul(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
            # 将每个Floor的数据连在一起
            RF_decoding_output = tf.concat([RF_decoding_L1_F1, RF_decoding_L1_F2, RF_decoding_L1_F3, RF_decoding_L1_F4,
                                            RF_decoding_L1_F5, RF_decoding_L1_F6, RF_decoding_L1_F7, RF_decoding_L1_F8
                                            ], 1)  # 两层

        with tf.name_scope('Baseband_decoding'):
            with tf.name_scope('Real_Channel'):
                # real_L1
                real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='real_decoding_L1')
                # real_L2
                real_decoding_L2_f = slim.fully_connected(real_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='real_decoding_L2')
                # real_L3
                real_decoding_L3_f = slim.fully_connected(real_decoding_L2_f, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='real_decoding_L3')
                # real_L4
                real_decoding_L4_f = slim.fully_connected(real_decoding_L3_f, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='real_decoding_L4')
                # real_L5
                real_decoding_L5_f = slim.fully_connected(real_decoding_L4_f, self.baseband_decoding_L5, activation_fn=self.activation_function, scope='real_decoding_L5')
                # real_L6
                real_decoding_L6_f = slim.fully_connected(real_decoding_L5_f, self.baseband_decoding_L6, activation_fn=self.activation_function, scope='real_decoding_L6')
                # real_L7
                real_decoding_L7_f = slim.fully_connected(real_decoding_L6_f, self.baseband_decoding_L7, activation_fn=self.activation_function, scope='real_decoding_L7')
                # real_L8
                real_decoding_L8_f = slim.fully_connected(real_decoding_L7_f, self.baseband_decoding_L8, activation_fn=self.activation_function, scope='real_decoding_L8')
                # real_L9
                real_decoding_L9_f = slim.fully_connected(real_decoding_L8_f, self.baseband_decoding_L9, activation_fn=None, scope='real_decoding_L9')
                real_prediction = real_decoding_L9_f
                # print("real_prediction:\n",real_prediction)
            with tf.name_scope('Imag_Channel'):
                # imag_L1
                imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), self.baseband_decoding_L1, activation_fn=self.activation_function, scope='imag_decoding_L1')
                # imag_L2
                imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1_f, self.baseband_decoding_L2, activation_fn=self.activation_function, scope='imag_decoding_L2')
                # imag_L3
                imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2_f, self.baseband_decoding_L3, activation_fn=self.activation_function, scope='imag_decoding_L3')
                # imag_L4
                imag_decoding_L4_f = slim.fully_connected(imag_decoding_L3_f, self.baseband_decoding_L4, activation_fn=self.activation_function, scope='imag_decoding_L4')
                # imag_L5
                imag_decoding_L5_f = slim.fully_connected(imag_decoding_L4_f, self.baseband_decoding_L5, activation_fn=self.activation_function, scope='imag_decoding_L5')
                # imag_L6
                imag_decoding_L6_f = slim.fully_connected(imag_decoding_L5_f, self.baseband_decoding_L6, activation_fn=self.activation_function, scope='imag_decoding_L6')
                # imag_L7
                imag_decoding_L7_f = slim.fully_connected(imag_decoding_L6_f, self.baseband_decoding_L7, activation_fn=self.activation_function, scope='imag_decoding_L7')
                # imag_L8
                imag_decoding_L8_f = slim.fully_connected(imag_decoding_L7_f, self.baseband_decoding_L8, activation_fn=self.activation_function, scope='imag_decoding_L8')
                # imag_L9
                imag_decoding_L9_f = slim.fully_connected(imag_decoding_L8_f, self.baseband_decoding_L9, activation_fn=None, scope='imag_decoding_L9')
                imag_prediction = imag_decoding_L9_f
                # print("imag_prediction:\n",imag_prediction)
            output = tf.complex(real_prediction, imag_prediction, name="output")

        with tf.name_scope('Loss'):
            # loss function
            self.loss = tf.reduce_mean(tf.square(tf.real(self.x) - real_prediction) + tf.square(tf.imag(self.x) - imag_prediction))

        # 通知 tensorflow 在训练时要更新均值的方差的分布
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.Learning_rate).minimize(self.loss)

        self.predictions["baseband_precoding_output"] = baseband_precoding_output
        self.predictions["RF_precoding_output"] = RF_precoding_output
        self.predictions["power_constrained_output"] = power_constrained_output
        self.predictions["H_output"] = H_output
        self.predictions["noise"] = noise
        self.predictions["noise_output"] = noise_output
        self.predictions["RF_decoding_output"] = RF_decoding_output
        self.predictions["output"] = output

        # 需要保存的中间参数
        tf.summary.scalar('baseband_precoding_output', signal.signal_power_tf(self.predictions["baseband_precoding_output"]))
        tf.summary.scalar('RF_precoding_output', signal.signal_power_tf(self.predictions["RF_precoding_output"]))
        tf.summary.scalar('power_constrained_output', signal.signal_power_tf(self.predictions["power_constrained_output"]))
        tf.summary.scalar('H_output', signal.signal_power_tf(self.predictions["H_output"]))
        tf.summary.scalar('noise', signal.signal_power_tf(self.predictions["noise"]))
        tf.summary.scalar('noise_output', signal.signal_power_tf(self.predictions["noise_output"]))
        tf.summary.scalar('RF_decoding_output', signal.signal_power_tf(self.predictions["RF_decoding_output"]))
        tf.summary.scalar('output', signal.signal_power_tf(self.predictions["output"]))
        tf.summary.scalar('Loss', self.loss)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        return self.predictions, self.loss, self.train_step, self.merged

