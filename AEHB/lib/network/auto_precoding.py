import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import lib.config.config as cfg
import lib.utils.beamforming_tools as bt
import lib.utils.SaveAndLoad as sal
import lib.utils.communication_tools as ct
import os

signal = ct.signal(cfg.FLAGS.modulation_mode, cfg.FLAGS.power_normalization, cfg.FLAGS.Ns)


def precoding(datasets):

    #######################
    # parameter definition
    n_training = cfg.FLAGS.N_training
    n_frame_training = cfg.FLAGS.N_frame_training
    N_frame_test = int(cfg.FLAGS.N_bits_test/cfg.FLAGS.Ns/signal.bit)
    batch_size = cfg.FLAGS.batch_size

    # 激活函数
    # activation_function = tf.nn.softsign
    # activation_function = tf.nn.softplus
    # activation_function = tf.nn.sigmoid
    # activation_function = tf.nn.tanh
    activation_function = tf.nn.relu
    # activation_function = tf.nn.relu6
    # activation_function = tf.nn.elu

    # 层数定义
    Ns = cfg.FLAGS.Ns
    Nr = cfg.FLAGS.Nr
    Nt = cfg.FLAGS.Nt
    Ntrf = cfg.FLAGS.Ntrf
    Nrrf = cfg.FLAGS.Nrrf
    N = 4
    baseband_precoding_L0 = cfg.FLAGS.Ns
    baseband_precoding_L1 = cfg.FLAGS.Ns * 2 * N
    baseband_precoding_L2 = cfg.FLAGS.Ns * 4 * N
    baseband_precoding_L3 = cfg.FLAGS.Ns * 4 * N
    baseband_precoding_L4 = cfg.FLAGS.Ns * 2 * N
    baseband_precoding_L5 = cfg.FLAGS.Ntrf
    RF_precoding_L1 = cfg.FLAGS.Nt
    RF_decoding_L1 = cfg.FLAGS.Nr
    baseband_decoding_L0 = cfg.FLAGS.Nrrf
    baseband_decoding_L1 = cfg.FLAGS.Ns * 2 * N
    baseband_decoding_L2 = cfg.FLAGS.Ns * 4 * N
    baseband_decoding_L3 = cfg.FLAGS.Ns * 4 * N
    baseband_decoding_L4 = cfg.FLAGS.Ns * 2 * N
    baseband_decoding_L5 = cfg.FLAGS.Ns
    ##########################################

    def simulation():
        results = {}
        prediction_value = {}
        step_value = []
        loss_value = []
        acc_value = []
        step_for_acc = []

        # 动态显示
        plt.figure(1)
        plt.ion()
        n_epoch = n_training//(n_frame_training // batch_size)
        for epoch in range(n_epoch):
            # 设置Learning_rate 可变
            if epoch < 0.03 * n_epoch:
                learning_rate = 0.000001
            elif epoch < 0.90 * n_epoch:
                learning_rate = 0.001
            else:
                learning_rate = 0.000001
            # learning_rate = 0.001
            x_data_training = datasets["x_data_training"]
            np.random.shuffle(x_data_training)
            for step in range(n_frame_training // batch_size):
                n_step = epoch * (n_frame_training // batch_size) + step
                x_feed = x_data_training[step * batch_size:(step + 1) * batch_size, :]

                # 参照发射信号的能量调整噪声功率以满足信噪比公式
                # SNR = np.random.uniform(-10, 10)  # 训练噪声信噪比
                # SNR = 10
                noise_data = signal.noise_generator(1.0, SNR, (batch_size, cfg.FLAGS.Nr))  # 训练时产生大小变化的噪声
                summary, _, loss_temp, predictions_train = sess.run([merged, train_step, loss, predictions], feed_dict={x: x_feed, y: x_feed, H: datasets["H_data"], noise: noise_data, training: True, Learning_rate: learning_rate})
                train_writer.add_summary(summary, n_step)
                # 计算能量
                power_RF_output = np.sum(signal.signal_power(predictions_train["RF_precoding_output"], 0))
                power_constrained = np.sum(signal.signal_power(predictions_train["power_constrained_output"], 0))
                power_H_output = np.sum(signal.signal_power(predictions_train["H_output"], 0))
                power_baseband_precoding_output = np.sum(signal.signal_power(predictions_train["baseband_precoding_output"], 0))
                # 保存训练过程
                step_value.append(n_step)
                loss_value.append(loss_temp)
                if n_step % 1000 == 0:   # 测试
                    # 测试，计算准确率
                    noise_data_test = signal.noise_generator(power_constrained, cfg.FLAGS.SNR_test, (N_frame_test, cfg.FLAGS.Nr))  # 产生测试噪声
                    summary, prediction_value = sess.run([merged, predictions], feed_dict={x: datasets["x_data_test"], y: datasets["x_data_test"], H: datasets["H_data"], noise: noise_data_test, training: False})  # 测试
                    test_writer.add_summary(summary, n_step)
                    binary_prediction, _, _ = signal.signal_decoder(prediction_value["output"])
                    acc = signal.count_accuracy_rate(binary_prediction, datasets["binary_data_test"])
                    acc_value.append(acc)
                    step_for_acc.append(n_step)

                    # 输出
                    print("power_RF_output:", power_RF_output)
                    print("power_constrained:", power_constrained)
                    print("power_baseband_precoding_output:", power_baseband_precoding_output)
                    print("power_H_output:", power_H_output)
                    print("After", n_step, ", training loss=", loss_temp, ", Acc=", acc, ", Learning_rate=", learning_rate, ", training SNR=", str(SNR), ", test SNR=", str(cfg.FLAGS.SNR_test), "\n")

                    plt.cla()
                    plt.scatter(np.real(prediction_value["output"]), np.imag(prediction_value["output"]), s=10)
                    plt.scatter(np.real(datasets["scatter_std"]), np.imag(datasets["scatter_std"]), s=30)
                    plt.xlabel("Real")
                    plt.ylabel("Imag")
                    plt.title("scatter figure after " + str(n_step) + " trains,\nBER = " + str(1 - acc) + "\nSNR = " + str(cfg.FLAGS.SNR_test) + "dB")
                    plt.pause(0.01)
                if n_step % 1000 == 0:  # 保存模型
                    saver.save(sess, cfg.FLAGS.path_for_ckpt)

        # 保存仿真数据
        results["loss_x"] = step_value
        results["loss_y"] = loss_value
        results["acc_x"] = step_for_acc
        results["acc_y"] = acc_value
        results["scatter_data"] = (prediction_value["output"])
        results["scatter_std"] = datasets["scatter_std"]
        sal.save_pkl(cfg.FLAGS.path_for_results, results)  # 保存训练数据

        plt.ioff()
        plt.show()

        plt.figure(2)
        plt.title("curve of loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.plot(step_value, loss_value)
        plt.figure(3)
        plt.title("curve of acc")
        plt.xlabel("step")
        plt.ylabel("BER")
        plt.plot(step_for_acc, acc_value)
        plt.show()

    #######################
    # data set
    #######################
    predictions = {}

    # placeholder
    x = tf.placeholder(tf.complex64, [None, Ns], name="x")
    y = tf.placeholder(tf.complex64, [None, Ns], name="y")
    H = tf.placeholder(tf.complex64, [Nt, Nr], name="H")
    noise = tf.placeholder(tf.complex64, [None, Nr], name="noise")
    training = tf.placeholder(tf.bool, None, name="training")
    Learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate")

    # build network
    with tf.name_scope('Baseband_Precoding'):
        with tf.name_scope('Real_channel'):
            # real_L1
            real_L1_f = slim.fully_connected(tf.real(x), baseband_precoding_L1, activation_fn=activation_function, scope='real_L1')
            real_L1 = tf.layers.batch_normalization(real_L1_f, training=training)
            # real_L2
            real_L2_f = slim.fully_connected(real_L1, baseband_precoding_L2, activation_fn=activation_function, scope='real_L2')
            real_L2 = tf.layers.batch_normalization(real_L2_f, training=training)
            # real_L3
            real_L3_f = slim.fully_connected(real_L2, baseband_precoding_L3, activation_fn=activation_function, scope='real_L3')
            real_L3 = tf.layers.batch_normalization(real_L3_f, training=training)
            # real_L4
            real_L4_f = slim.fully_connected(real_L3, baseband_precoding_L4, activation_fn=activation_function, scope='real_L4')
            real_L4 = tf.layers.batch_normalization(real_L4_f, training=training)
            # real_L5
            real_L5_f = slim.fully_connected(real_L4, baseband_precoding_L5, activation_fn=activation_function, scope='real_L5')
            real_L5 = tf.layers.batch_normalization(real_L5_f, training=training)
            # real_baseband_output
            real_baseband_output = real_L5

        with tf.name_scope('Imag_channel'):
            # imag_L1
            imag_L1_f = slim.fully_connected(tf.imag(x), baseband_precoding_L1, activation_fn=activation_function, scope='imag_L1')
            imag_L1 = tf.layers.batch_normalization(imag_L1_f, training=training)
            # imag_L2
            imag_L2_f = slim.fully_connected(imag_L1, baseband_precoding_L2, activation_fn=activation_function, scope='imag_L2')
            imag_L2 = tf.layers.batch_normalization(imag_L2_f, training=training)
            # imag_L3
            imag_L3_f = slim.fully_connected(imag_L2, baseband_precoding_L3, activation_fn=activation_function, scope='imag_L3')
            imag_L3 = tf.layers.batch_normalization(imag_L3_f, training=training)
            # imag_L4
            imag_L4_f = slim.fully_connected(imag_L3, baseband_precoding_L4, activation_fn=activation_function, scope='imag_L4')
            imag_L4 = tf.layers.batch_normalization(imag_L4_f, training=training)
            # imag_L5
            imag_L5_f = slim.fully_connected(imag_L4, baseband_precoding_L5, activation_fn=activation_function, scope='imag_L5')
            imag_L5 = tf.layers.batch_normalization(imag_L5_f, training=training)
            # imag_baseband_output
            imag_baseband_output = imag_L5
        # baseband_precoding_output
        baseband_precoding_output = tf.complex(real_baseband_output, imag_baseband_output, name="baseband_output")

    with tf.name_scope('RF_Precoding'):
        n_p1 = tf.to_int32(RF_precoding_L1 / Ntrf)
        theta_precoding_L1 = tf.Variable(tf.random_uniform([Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L1")
        theta_precoding_L2 = tf.Variable(tf.random_uniform([Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L2")
        theta_precoding_L3 = tf.Variable(tf.random_uniform([Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L3")
        theta_precoding_L4 = tf.Variable(tf.random_uniform([Ntrf, n_p1], minval=0, maxval=2 * np.pi), name="theta_precoding_L4")
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
        with tf.name_scope('Layer_2'):
            # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
            # L2_Floor1
            RF_precoding_L2_F1 = bt.phase_shift_multiply(RF_precoding_L1_F1, tf.reshape(theta_precoding_L2[0, :], (-1, n_p1)), name="RF_L2_F1")
            # L2_Floor2
            RF_precoding_L2_F2 = bt.phase_shift_multiply(RF_precoding_L1_F2, tf.reshape(theta_precoding_L2[1, :], (-1, n_p1)), name="RF_L2_F2")
            # L2_Floor3
            RF_precoding_L2_F3 = bt.phase_shift_multiply(RF_precoding_L1_F3, tf.reshape(theta_precoding_L2[2, :], (-1, n_p1)), name="RF_L2_F3")
            # L2_Floor4
            RF_precoding_L2_F4 = bt.phase_shift_multiply(RF_precoding_L1_F4, tf.reshape(theta_precoding_L2[3, :], (-1, n_p1)), name="RF_L2_F4")
            # L2_Floor5
            RF_precoding_L2_F5 = bt.phase_shift_multiply(RF_precoding_L1_F5, tf.reshape(theta_precoding_L2[4, :], (-1, n_p1)), name="RF_L2_F5")
            # L2_Floor6
            RF_precoding_L2_F6 = bt.phase_shift_multiply(RF_precoding_L1_F6, tf.reshape(theta_precoding_L2[5, :], (-1, n_p1)), name="RF_L2_F6")
            # L2_Floor7
            RF_precoding_L2_F7 = bt.phase_shift_multiply(RF_precoding_L1_F7, tf.reshape(theta_precoding_L2[6, :], (-1, n_p1)), name="RF_L2_F7")
            # L2_Floor8
            RF_precoding_L2_F8 = bt.phase_shift_multiply(RF_precoding_L1_F8, tf.reshape(theta_precoding_L2[7, :], (-1, n_p1)), name="RF_L2_F8")
        with tf.name_scope('Layer_3'):
            # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
            # L3_Floor1
            RF_precoding_L3_F1 = bt.phase_shift_multiply(RF_precoding_L2_F1, tf.reshape(theta_precoding_L3[0, :], (-1, n_p1)), name="RF_L3_F1")
            # L3_Floor2
            RF_precoding_L3_F2 = bt.phase_shift_multiply(RF_precoding_L2_F2, tf.reshape(theta_precoding_L3[1, :], (-1, n_p1)), name="RF_L3_F2")
            # L3_Floor3
            RF_precoding_L3_F3 = bt.phase_shift_multiply(RF_precoding_L2_F3, tf.reshape(theta_precoding_L3[2, :], (-1, n_p1)), name="RF_L3_F3")
            # L3_Floor4
            RF_precoding_L3_F4 = bt.phase_shift_multiply(RF_precoding_L2_F4, tf.reshape(theta_precoding_L3[3, :], (-1, n_p1)), name="RF_L3_F4")
            # L3_Floor5
            RF_precoding_L3_F5 = bt.phase_shift_multiply(RF_precoding_L2_F5, tf.reshape(theta_precoding_L3[4, :], (-1, n_p1)), name="RF_L3_F5")
            # L3_Floor6
            RF_precoding_L3_F6 = bt.phase_shift_multiply(RF_precoding_L2_F6, tf.reshape(theta_precoding_L3[5, :], (-1, n_p1)), name="RF_L3_F6")
            # L3_Floor7
            RF_precoding_L3_F7 = bt.phase_shift_multiply(RF_precoding_L2_F7, tf.reshape(theta_precoding_L3[6, :], (-1, n_p1)), name="RF_L3_F7")
            # L3_Floor8
            RF_precoding_L3_F8 = bt.phase_shift_multiply(RF_precoding_L2_F8, tf.reshape(theta_precoding_L3[7, :], (-1, n_p1)), name="RF_L3_F8")
        with tf.name_scope('Layer_4'):
            # 共有Ntrf个floor，修改Ntrf的时候，记得修改下列代码
            # L4_Floor1
            RF_precoding_L4_F1 = bt.phase_shift_multiply(RF_precoding_L3_F1, tf.reshape(theta_precoding_L4[0, :], (-1, n_p1)), name="RF_L4_F1")
            # L4_Floor2
            RF_precoding_L4_F2 = bt.phase_shift_multiply(RF_precoding_L3_F2, tf.reshape(theta_precoding_L4[1, :], (-1, n_p1)), name="RF_L4_F2")
            # L4_Floor3
            RF_precoding_L4_F3 = bt.phase_shift_multiply(RF_precoding_L3_F3, tf.reshape(theta_precoding_L4[2, :], (-1, n_p1)), name="RF_L4_F3")
            # L4_Floor4
            RF_precoding_L4_F4 = bt.phase_shift_multiply(RF_precoding_L3_F4, tf.reshape(theta_precoding_L4[3, :], (-1, n_p1)), name="RF_L4_F4")
            # L4_Floor5
            RF_precoding_L4_F5 = bt.phase_shift_multiply(RF_precoding_L3_F5, tf.reshape(theta_precoding_L4[4, :], (-1, n_p1)), name="RF_L4_F5")
            # L4_Floor6
            RF_precoding_L4_F6 = bt.phase_shift_multiply(RF_precoding_L3_F6, tf.reshape(theta_precoding_L4[5, :], (-1, n_p1)), name="RF_L4_F6")
            # L4_Floor7
            RF_precoding_L4_F7 = bt.phase_shift_multiply(RF_precoding_L3_F7, tf.reshape(theta_precoding_L4[6, :], (-1, n_p1)), name="RF_L4_F7")
            # L4_Floor8
            RF_precoding_L4_F8 = bt.phase_shift_multiply(RF_precoding_L3_F8, tf.reshape(theta_precoding_L4[7, :], (-1, n_p1)), name="RF_L4_F8")
        # 将每个Floor的数据连在一起
        RF_precoding_output = tf.concat([RF_precoding_L4_F1, RF_precoding_L4_F2, RF_precoding_L4_F3, RF_precoding_L4_F4,
                                         RF_precoding_L4_F5, RF_precoding_L4_F6, RF_precoding_L4_F7, RF_precoding_L4_F8
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
        n_d1 = tf.to_int32(RF_decoding_L1 / Nrrf)
        theta_decoding_L1 = tf.Variable(tf.random_uniform([Nrrf, n_d1], minval=0, maxval=2 * np.pi), name="theta_decoding_L1")
        theta_decoding_L2 = tf.Variable(tf.random_uniform([Nrrf, n_d1], minval=0, maxval=2 * np.pi), name="theta_decoding_L2")
        theta_decoding_L3 = tf.Variable(tf.random_uniform([Nrrf, n_d1], minval=0, maxval=2 * np.pi), name="theta_decoding_L3")
        theta_decoding_L4 = tf.Variable(tf.random_uniform([n_d1, Nrrf], minval=0, maxval=2 * np.pi), name="theta_decoding_L4")
        with tf.name_scope('Layer_1'):
            # 共有Nrrf个floor，修改Nrrf的时候，记得修改下列代码
            # L1_Floor1
            RF_decoding_L1_F1 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 0 * n_d1:1 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[0, :], (-1, n_d1)), name="RF_decoding_L1_F1")
            # L1_Floor2
            RF_decoding_L1_F2 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 1 * n_d1:2 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[1, :], (-1, n_d1)), name="RF_decoding_L1_F2")
            # L1_Floor3
            RF_decoding_L1_F3 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 2 * n_d1:3 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[2, :], (-1, n_d1)), name="RF_decoding_L1_F3")
            # L1_Floor4
            RF_decoding_L1_F4 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 3 * n_d1:4 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[3, :], (-1, n_d1)), name="RF_decoding_L1_F4")
            # L1_Floor5
            RF_decoding_L1_F5 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 4 * n_d1:5 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[4, :], (-1, n_d1)), name="RF_decoding_L1_F5")
            # L1_Floor6
            RF_decoding_L1_F6 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 5 * n_d1:6 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[5, :], (-1, n_d1)), name="RF_decoding_L1_F6")
            # L1_Floor7
            RF_decoding_L1_F7 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 6 * n_d1:7 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[6, :], (-1, n_d1)), name="RF_decoding_L1_F7")
            # L1_Floor8
            RF_decoding_L1_F8 = bt.phase_shift_multiply(tf.reshape(noise_output[:, 7 * n_d1:8 * n_d1], (-1, n_d1)), tf.reshape(theta_decoding_L1[7, :], (-1, n_d1)), name="RF_decoding_L1_F8")
        with tf.name_scope('Layer_2'):
            # L2_Floor1
            RF_decoding_L2_F1 = bt.phase_shift_multiply(RF_decoding_L1_F1, tf.reshape(theta_decoding_L2[0, :], (-1, n_d1)), name="RF_decoding_L2_F1")
            # L2_Floor2
            RF_decoding_L2_F2 = bt.phase_shift_multiply(RF_decoding_L1_F2, tf.reshape(theta_decoding_L2[1, :], (-1, n_d1)), name="RF_decoding_L2_F2")
            # L2_Floor3
            RF_decoding_L2_F3 = bt.phase_shift_multiply(RF_decoding_L1_F3, tf.reshape(theta_decoding_L2[2, :], (-1, n_d1)), name="RF_decoding_L2_F3")
            # L2_Floor4
            RF_decoding_L2_F4 = bt.phase_shift_multiply(RF_decoding_L1_F4, tf.reshape(theta_decoding_L2[3, :], (-1, n_d1)), name="RF_decoding_L2_F4")
            # L2_Floor5
            RF_decoding_L2_F5 = bt.phase_shift_multiply(RF_decoding_L1_F5, tf.reshape(theta_decoding_L2[4, :], (-1, n_d1)), name="RF_decoding_L2_F5")
            # L2_Floor6
            RF_decoding_L2_F6 = bt.phase_shift_multiply(RF_decoding_L1_F6, tf.reshape(theta_decoding_L2[5, :], (-1, n_d1)), name="RF_decoding_L2_F6")
            # L2_Floor7
            RF_decoding_L2_F7 = bt.phase_shift_multiply(RF_decoding_L1_F7, tf.reshape(theta_decoding_L2[6, :], (-1, n_d1)), name="RF_decoding_L2_F7")
            # L2_Floor8
            RF_decoding_L2_F8 = bt.phase_shift_multiply(RF_decoding_L1_F8, tf.reshape(theta_decoding_L2[7, :], (-1, n_d1)), name="RF_decoding_L2_F8")
        with tf.name_scope('Layer_3'):
            # L2_Floor1
            RF_decoding_L3_F1 = bt.phase_shift_multiply(RF_decoding_L2_F1, tf.reshape(theta_decoding_L3[0, :], (-1, n_d1)), name="RF_decoding_L3_F1")
            # L2_Floor2
            RF_decoding_L3_F2 = bt.phase_shift_multiply(RF_decoding_L2_F2, tf.reshape(theta_decoding_L3[1, :], (-1, n_d1)), name="RF_decoding_L3_F2")
            # L2_Floor3
            RF_decoding_L3_F3 = bt.phase_shift_multiply(RF_decoding_L2_F3, tf.reshape(theta_decoding_L3[2, :], (-1, n_d1)), name="RF_decoding_L3_F3")
            # L2_Floor4
            RF_decoding_L3_F4 = bt.phase_shift_multiply(RF_decoding_L2_F4, tf.reshape(theta_decoding_L3[3, :], (-1, n_d1)), name="RF_decoding_L3_F4")
            # L2_Floor5
            RF_decoding_L3_F5 = bt.phase_shift_multiply(RF_decoding_L2_F5, tf.reshape(theta_decoding_L3[4, :], (-1, n_d1)), name="RF_decoding_L3_F5")
            # L2_Floor6
            RF_decoding_L3_F6 = bt.phase_shift_multiply(RF_decoding_L2_F6, tf.reshape(theta_decoding_L3[5, :], (-1, n_d1)), name="RF_decoding_L3_F6")
            # L2_Floor7
            RF_decoding_L3_F7 = bt.phase_shift_multiply(RF_decoding_L2_F7, tf.reshape(theta_decoding_L3[6, :], (-1, n_d1)), name="RF_decoding_L3_F7")
            # L2_Floor8
            RF_decoding_L3_F8 = bt.phase_shift_multiply(RF_decoding_L2_F8, tf.reshape(theta_decoding_L3[7, :], (-1, n_d1)), name="RF_decoding_L3_F8")
        with tf.name_scope('Layer_4'):
            # L2_Floor1
            RF_decoding_L4_F1 = bt.phase_shift_matmul(RF_decoding_L3_F1, tf.reshape(theta_decoding_L4[:, 0], (-1, 1)), name="RF_decoding_L4_F1")
            # L2_Floor2
            RF_decoding_L4_F2 = bt.phase_shift_matmul(RF_decoding_L3_F2, tf.reshape(theta_decoding_L4[:, 1], (-1, 1)), name="RF_decoding_L4_F2")
            # L2_Floor3
            RF_decoding_L4_F3 = bt.phase_shift_matmul(RF_decoding_L3_F3, tf.reshape(theta_decoding_L4[:, 2], (-1, 1)), name="RF_decoding_L4_F3")
            # L2_Floor4
            RF_decoding_L4_F4 = bt.phase_shift_matmul(RF_decoding_L3_F4, tf.reshape(theta_decoding_L4[:, 3], (-1, 1)), name="RF_decoding_L4_F4")
            # L2_Floor5
            RF_decoding_L4_F5 = bt.phase_shift_matmul(RF_decoding_L3_F5, tf.reshape(theta_decoding_L4[:, 4], (-1, 1)), name="RF_decoding_L4_F5")
            # L2_Floor6
            RF_decoding_L4_F6 = bt.phase_shift_matmul(RF_decoding_L3_F6, tf.reshape(theta_decoding_L4[:, 5], (-1, 1)), name="RF_decoding_L4_F6")
            # L2_Floor7
            RF_decoding_L4_F7 = bt.phase_shift_matmul(RF_decoding_L3_F7, tf.reshape(theta_decoding_L4[:, 6], (-1, 1)), name="RF_decoding_L4_F7")
            # L2_Floor8
            RF_decoding_L4_F8 = bt.phase_shift_matmul(RF_decoding_L3_F8, tf.reshape(theta_decoding_L4[:, 7], (-1, 1)), name="RF_decoding_L4_F8")
        # 将每个Floor的数据连在一起
        RF_decoding_output = tf.concat([RF_decoding_L4_F1, RF_decoding_L4_F2, RF_decoding_L4_F3, RF_decoding_L4_F4,
                                        RF_decoding_L4_F5, RF_decoding_L4_F6, RF_decoding_L4_F7, RF_decoding_L4_F8
                                        ], 1)  # 两层

    with tf.name_scope('Baseband_decoding'):
        with tf.name_scope('Real_Channel'):
            # real_L1
            real_decoding_L1_f = slim.fully_connected(tf.real(RF_decoding_output), baseband_decoding_L1, activation_fn=activation_function, scope='real_decoding__L1')
            real_decoding_L1 = tf.layers.batch_normalization(real_decoding_L1_f, training=training)
            # real_L2
            real_decoding_L2_f = slim.fully_connected(real_decoding_L1, baseband_decoding_L2, activation_fn=activation_function, scope='real_decoding__L2')
            real_decoding_L2 = tf.layers.batch_normalization(real_decoding_L2_f, training=training)
            # real_L3
            real_decoding_L3_f = slim.fully_connected(real_decoding_L2, baseband_decoding_L3, activation_fn=activation_function, scope='real_decoding__L3')
            real_decoding_L3 = tf.layers.batch_normalization(real_decoding_L3_f, training=training)
            # real_L4
            real_decoding_L4_f = slim.fully_connected(real_decoding_L3, baseband_decoding_L4, activation_fn=activation_function, scope='real_decoding__L4')
            real_decoding_L4 = tf.layers.batch_normalization(real_decoding_L4_f, training=training)
            # real_L5
            real_decoding_L5_f = slim.fully_connected(real_decoding_L4, baseband_decoding_L5, activation_fn=None, scope='real_decoding__L5')
            real_prediction = real_decoding_L5_f
            # print("real_prediction:\n",real_prediction)
        with tf.name_scope('Imag_Channel'):
            # imag_L1
            imag_decoding_L1_f = slim.fully_connected(tf.imag(RF_decoding_output), baseband_decoding_L1, activation_fn=activation_function, scope='imag_decoding__L1')
            imag_decoding_L1 = tf.layers.batch_normalization(imag_decoding_L1_f, training=training)
            # imag_L2
            imag_decoding_L2_f = slim.fully_connected(imag_decoding_L1, baseband_decoding_L2, activation_fn=activation_function, scope='imag_decoding__L2')
            imag_decoding_L2 = tf.layers.batch_normalization(imag_decoding_L2_f, training=training)
            # imag_L3
            imag_decoding_L3_f = slim.fully_connected(imag_decoding_L2, baseband_decoding_L3, activation_fn=activation_function, scope='imag_decoding__L3')
            imag_decoding_L3 = tf.layers.batch_normalization(imag_decoding_L3_f, training=training)
            # imag_L4
            imag_decoding_L4_f = slim.fully_connected(imag_decoding_L3, baseband_decoding_L4, activation_fn=activation_function, scope='imag_decoding__L4')
            imag_decoding_L4 = tf.layers.batch_normalization(imag_decoding_L4_f, training=training)
            # imag_L5
            imag_decoding_L5_f = slim.fully_connected(imag_decoding_L4, baseband_decoding_L5, activation_fn=None, scope='imag_decoding__L5')
            imag_prediction = imag_decoding_L5_f
            # print("imag_prediction:\n",imag_prediction)
        output = tf.complex(real_prediction, imag_prediction, name="output")

    predictions["baseband_precoding_output"] = baseband_precoding_output
    predictions["RF_precoding_output"] = RF_precoding_output
    predictions["power_constrained_output"] = power_constrained_output
    predictions["H_output"] = H_output
    predictions["noise_output"] = noise_output
    predictions["RF_decoding_output"] = RF_decoding_output
    predictions["output"] = output

    with tf.name_scope('Loss'):
        # loss function
        loss = tf.reduce_mean(tf.square(tf.real(y) - real_prediction) + tf.square(tf.imag(y) - imag_prediction))

    # 需要保存的中间参数
    tf.summary.scalar('baseband_precoding_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(baseband_precoding_output), 2), 0)))
    tf.summary.scalar('RF_precoding_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(RF_precoding_output), 2), 0)))
    tf.summary.scalar('power_constrained_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(power_constrained_output), 2), 0)))
    tf.summary.scalar('H_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(H_output), 2), 0)))
    tf.summary.scalar('noise_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(noise_output), 2), 0)))
    tf.summary.scalar('RF_decoding_output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(RF_decoding_output), 2), 0)))
    tf.summary.scalar('output', tf.reduce_sum(tf.reduce_mean(tf.pow(tf.abs(output), 2), 0)))
    tf.summary.scalar('Loss', loss)

    # 通知 tensorflow 在训练时要更新均值的方差的分布
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    # variable initializer
    init = tf.global_variables_initializer()

    # 创建一个实例
    saver = tf.train.Saver()
    # 判断是否存在已训练的模型
    if os.access(cfg.FLAGS.path_for_ckpt + ".meta", os.F_OK):
        print("The model exists, and the training will continue on the original basis!\n")
        with tf.Session() as sess:
            # load meta graph and restore weights
            saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/train',
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/test')
            saver.restore(sess, cfg.FLAGS.path_for_ckpt)
            simulation()
    else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/train',
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/test')

            sess.run(init)
            simulation()
    return

# Fbb, Frf, W, H_data = precoding()
# print("Fbb:\n", Fbb)
# print("Frf:\n", Frf)
# print("W:\n", W)
# print("H_data:\n", H_data)
