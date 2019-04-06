import torch
import torch.nn.functional as F

# ******** CRF 工具函数*************


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


# ******** LSTM模型 工具函数*************

def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def cal_loss(logits, targets, tag2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L]
        lengths: [B]
    """
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss

# FOR BiLSTM-CRF


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    # 计算golden_scores方法２：利用pack_padded_sequence函数
    # targets[targets == end_id] = pad_id
    # scores_at_targets = torch.gather(
    #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
    # scores_at_targets, _ = pack_padded_sequence(
    #     scores_at_targets, lengths-1, batch_first=True
    # )
    # golden_scores = scores_at_targets.sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets
