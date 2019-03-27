import time
from collections import Counter

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists


def evaluate(tag_lists, target_tag_lists):
    # 评估准确率
    correct_count = 0.

    # 展开嵌套列表
    tag_lists = flatten_lists(tag_lists)
    target_tag_lists = flatten_lists(target_tag_lists)
    assert len(tag_lists) == len(target_tag_lists)

    for pred, tgt in zip(tag_lists, target_tag_lists):
        if pred == tgt:
            correct_count += 1.
    return correct_count/len(tag_lists)


def hmm_train_eval(train_data, test_data, word2id, tag2id):
    """训练并评估hmm模型"""
    # 训练HMM模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)
    accuracy = evaluate(pred_tag_lists, test_tag_lists)
    print("HMM 模型的准确率为：{:.2f}%".format(accuracy * 100))

    return pred_tag_lists


def crf_train_eval(train_data, test_data):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)
    accuracy = evaluate(pred_tag_lists, test_tag_lists)
    print("CRF 模型的准确率为：{:.2f}%".format(accuracy * 100))

    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    accuracy = evaluate(pred_tag_lists, test_tag_lists)
    print("{} 模型的准确率为：{:.2f}%".format(
        model_name, accuracy * 100
    ))

    return pred_tag_lists


def ensemble_evaluate(results, targets):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    correct = 0
    for pred, tgt in zip(pred_tags, targets):
        if pred == tgt:
            correct += 1.
    accuracy = correct/len(targets)

    print("Ensemble四个模型的准确率为{:.2f}%".format(accuracy * 100))
