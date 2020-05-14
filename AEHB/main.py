import lib.utils.communication_tools as ct
import lib.utils.SaveAndLoad as sal
from lib.network.precoding import precoding
import lib.config.config as cfg
import scipy.io as sio
import os

cfg.FLAGS.path = cfg.FLAGS.path + "/data/random1/" + cfg.FLAGS.modulation_mode
sal.mkdir(cfg.FLAGS.path)  # 创建文件夹
cfg.FLAGS.path_for_graph = cfg.FLAGS.path + "/graph"
cfg.FLAGS.path_for_datasets = cfg.FLAGS.path + "/datasets.pkl"
cfg.FLAGS.path_for_ckpt = cfg.FLAGS.path + "/constrained_DNN.ckpt"
cfg.FLAGS.path_for_results = cfg.FLAGS.path + "/results"
cfg.FLAGS.path_for_BER = cfg.FLAGS.path + "/BER"

signal = ct.signal(cfg.FLAGS.modulation_mode, cfg.FLAGS.power_normalization, cfg.FLAGS.Ns)
N_frame_test = int(cfg.FLAGS.N_bits_test / cfg.FLAGS.Ns / signal.bit)

# 创建训练与测试的数据集
if os.access(cfg.FLAGS.path_for_datasets, os.F_OK):
    print("loading datasets!")
    datasets = sal.load_pkl(cfg.FLAGS.path_for_datasets)

else:
    print("generate datasets for training and testing!")
    datasets = {}
    # training data
    _, datasets["x_data_training"], _, datasets["scatter_std"] = signal.signal_generator([cfg.FLAGS.N_frame_training, cfg.FLAGS.Ns])
    # test data
    datasets["binary_data_test"], datasets["x_data_test"], _, _, = signal.signal_generator([N_frame_test, cfg.FLAGS.Ns])

    # 创建随机信道矩阵
    # np.random.seed(1024)
    # H_data = np.random.rand(cfg.FLAGS.Nt, cfg.FLAGS.Nr) + np.random.rand(cfg.FLAGS.Nt, cfg.FLAGS.Nr) * 1j
    # datasets["H_data"] = H_data.astype(np.complex64)

    # 从.mat 文件中读取信道矩阵
    mimochannel = sio.loadmat("matlab_support/mimochannel.mat")
    datasets["H_data"] = mimochannel["mimochannel_equal"]
    sal.save_pkl(cfg.FLAGS.path_for_datasets, datasets)


precoding(datasets)