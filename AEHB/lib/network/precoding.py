import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import lib.config.config as cfg
import lib.network.net as net
import lib.utils.SaveAndLoad as sal
import lib.utils.communication_tools as ct
import os

signal = ct.signal(cfg.FLAGS.modulation_mode, cfg.FLAGS.power_normalization, cfg.FLAGS.Ns)


def precoding(datasets):
    #######################
    # parameter definition
    Ns = cfg.FLAGS.Ns
    Nr = cfg.FLAGS.Nr
    Nt = cfg.FLAGS.Nt

    n_training = cfg.FLAGS.N_training
    n_frame_training = cfg.FLAGS.N_frame_training
    N_frame_test = int(cfg.FLAGS.N_bits_test / cfg.FLAGS.Ns / signal.bit)
    batch_size = cfg.FLAGS.batch_size

    def simulation():
        results = {}
        prediction_value = {}
        step_value = []
        loss_value = []
        acc_value = []
        step_for_acc = []

        # 动态显示
        # plt.figure(1)
        # plt.ion()
        n_epoch = n_training // (n_frame_training // batch_size)
        SNR_index = 0
        for epoch in range(n_epoch):
            # 设置Learning_rate 可变
            if epoch < 0.03 * n_epoch:
                learning_rate = 0.000001
            elif epoch < 0.85 * n_epoch:
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
                # SNR_range = np.linspace(-10, 10, 21)  # 训练噪声信噪比
                # SNR_range = np.linspace(10, -10, 21)  # 训练噪声信噪比

                # if n_step % (np.int(n_training / 20)) == 0 :
                #     SNR_training = SNR_range[SNR_index]
                #     SNR_index = SNR_index + 1

                SNR_training = np.random.uniform(-10, 10)  # 训练噪声信噪比
                # SNR_training = cfg.FLAGS.SNR_training
                noise_data = signal.noise_generator(1.0, SNR_training, (batch_size, cfg.FLAGS.Nr))  # 训练时产生大小变化的噪声
                summary, _, loss_temp, predictions_train = sess.run([merged, train_step, loss, predictions],
                                                                    feed_dict={x: x_feed, H: datasets["H_data"], noise: noise_data, training: True, Learning_rate: learning_rate})
                train_writer.add_summary(summary, n_step)
                # 计算能量
                power_RF_output = np.sum(signal.signal_power(predictions_train["RF_precoding_output"], 0))
                power_constrained = np.sum(signal.signal_power(predictions_train["power_constrained_output"], 0))
                power_H_output = np.sum(signal.signal_power(predictions_train["H_output"], 0))
                power_baseband_precoding_output = np.sum(signal.signal_power(predictions_train["baseband_precoding_output"], 0))
                # 保存训练过程
                step_value.append(n_step)
                loss_value.append(loss_temp)
                if n_step % 100 == 0:  # 测试
                    # 测试，计算准确率
                    noise_data_test = signal.noise_generator(1.0, cfg.FLAGS.SNR_test, (N_frame_test, cfg.FLAGS.Nr))  # 产生测试噪声
                    summary, prediction_value = sess.run([merged, predictions],
                                                         feed_dict={x: datasets["x_data_test"], H: datasets["H_data"], noise: noise_data_test, training: False})  # 测试
                    # test_writer.add_summary(summary, n_step)
                    binary_prediction, _, _ = signal.signal_decoder(prediction_value["output"])
                    acc = signal.count_accuracy_rate(binary_prediction, datasets["binary_data_test"])
                    acc_value.append(acc)
                    step_for_acc.append(n_step)

                    # 输出
                    power_noise_data_test = signal.signal_power(noise_data_test)
                    print("power_noise_data_test", power_noise_data_test)
                    print("power_RF_output:", power_RF_output)
                    print("power_constrained:", power_constrained)
                    print("power_baseband_precoding_output:", power_baseband_precoding_output)
                    print("power_H_output:", power_H_output)
                    print("After", n_step, ", training loss=", loss_temp, ", Acc=", acc, ", Learning_rate=", learning_rate, ", training SNR=", str(SNR_training), ", test SNR=", str(cfg.FLAGS.SNR_test), "\n")

                    # plt.cla()
                    # plt.scatter(np.real(prediction_value["output"]), np.imag(prediction_value["output"]), s=10)
                    # plt.scatter(np.real(datasets["scatter_std"]), np.imag(datasets["scatter_std"]), s=30)
                    # plt.xlabel("Real")
                    # plt.ylabel("Imag")
                    # plt.title("scatter figure after " + str(n_step) + " trains,\nBER = " + str(1 - acc) + "\nSNR = " + str(cfg.FLAGS.SNR_test) + "dB")
                    # plt.pause(0.01)
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

    def ber_plot():
        BER = {}
        ber_x = []
        ber_y = []
        SNR_ber = np.array(np.linspace(-20, 15, 36))
        for _, snr in enumerate(SNR_ber):
            # 测试，计算准确率
            noise_data_test = signal.noise_generator(1.0, snr, (N_frame_test, cfg.FLAGS.Nr))  # 产生测试噪声
            summary, prediction_value = sess.run([merged, predictions],
                                                 feed_dict={x: datasets["x_data_test"], H: datasets["H_data"], noise: noise_data_test, training: False})  # 测试
            # test_writer.add_summary(summary, n_step)
            binary_prediction, _, _ = signal.signal_decoder(prediction_value["output"])
            acc = signal.count_accuracy_rate(binary_prediction, datasets["binary_data_test"])
            ber_y.append(1. - acc)
            print("SNR = " + str(snr) + ", ber = " + str(1. - acc))
            ber_x.append(snr)
        BER["ber_x"] = ber_x
        BER["ber_y"] = ber_y
        sal.save_pkl(cfg.FLAGS.path_for_BER, BER)

    # placeholder
    x = tf.placeholder(tf.complex64, [None, Ns], name="x")
    H = tf.placeholder(tf.complex64, [Nt, Nr], name="H")
    noise = tf.placeholder(tf.complex64, [None, Nr], name="noise")
    training = tf.placeholder(tf.bool, None, name="training")
    Learning_rate = tf.placeholder(tf.float32, shape=[], name="Learning_rate")

    # 创建神经网络
    cdnn = net.CdnnWithoutBN2Layers(tf.nn.tanh, 1)
    predictions, loss, train_step, merged = cdnn(x, H, noise, training, Learning_rate)

    # variable initializer
    init = tf.global_variables_initializer()

    # 创建一个实例
    # saver = tf.train.Saver()
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
            # simulation()
            ber_plot()
    else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/train',
                                                 sess.graph)
            # test_writer = tf.summary.FileWriter(cfg.FLAGS.path_for_graph + '/test')

            sess.run(init)
            simulation()
    return



