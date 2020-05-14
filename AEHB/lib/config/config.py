import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# simulation parameters
tf.app.flags.DEFINE_string('modulation_mode', "qpsk", 'modulation model')
tf.app.flags.DEFINE_string("power_normalization", "power_normalization_to_1/Ns", 'normalization mode')
tf.app.flags.DEFINE_integer('Ns', 4, 'number of data streams')
tf.app.flags.DEFINE_integer('Nt', 64, 'number of antenna at transmitter')
tf.app.flags.DEFINE_integer('Nr', 16, 'number of antenna at receiver')
tf.app.flags.DEFINE_integer('Ntrf', 8, 'number of RF chains at transmitter')
tf.app.flags.DEFINE_integer('Nrrf', 8, 'number of RF chains at receiver')

# model parameters
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_bool('constrained', True, 'True: power constrained; False without power constrained')
tf.app.flags.DEFINE_integer('N_training', 50000, "number of training times")
tf.app.flags.DEFINE_integer('N_frame_training', 10000, "number of frames during training")
tf.app.flags.DEFINE_integer('N_bits_test', 120000, "number of bits during test")
tf.app.flags.DEFINE_integer('N_iter', 1000, "number of iteration")
tf.app.flags.DEFINE_integer('SNR_test', 0, "SNR for test during training")
tf.app.flags.DEFINE_integer('SNR_training', 0, "SNR for training")
tf.app.flags.DEFINE_integer('batch_size', 128, "Network batch size during training")
tf.app.flags.DEFINE_integer('step_size', 10000, "Step size for reducing the learning rate, currently only support one step")

# file catalog
tf.app.flags.DEFINE_string("path", "H:/beamforming/cdnn/noise_test/cdnn_4_8_64_16_8_tanh", "当前工程的路径")
tf.app.flags.DEFINE_string('path_for_ckpt', "./data/modulation_mode/constrained_DNN.ckpt", 'path for training model of constrained_DNN')
tf.app.flags.DEFINE_string('path_for_graph', "./data/modulation_mode/graph", 'path for graph')
tf.app.flags.DEFINE_string('path_for_datasets', "./data/modulation_mode/datasets.npy", 'path for datasets')
tf.app.flags.DEFINE_string('path_for_results', "./data/modulation_mode/results.pkl", 'path for results')
tf.app.flags.DEFINE_string('path_for_BER', "./data/modulation_mode/BER", 'path for results')