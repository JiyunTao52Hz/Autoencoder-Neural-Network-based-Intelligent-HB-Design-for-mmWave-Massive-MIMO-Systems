import pickle
import os


def save_pkl(file_path, obj):
    """save obj to file_path"""
    output = open(file_path, 'wb')
    pickle.dump(obj, output)
    output.close()


def load_pkl(file_path):
    """load obj to file path"""
    pkl_file = open(file_path, 'rb')
    obj = pickle.load(pkl_file)
    return obj


def mkdir(path):
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)  # 创建目录操作函数
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

