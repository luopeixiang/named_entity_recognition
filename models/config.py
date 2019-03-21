# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 64
    # 学习速率
    lr = 0.01
    epoches = 60
    print_step = 10
    # 保存模型的的文件名
    save_file = "./ckpts/bilstm.pkl"


class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
