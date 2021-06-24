import pickle


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# 读取txt里的内容到list
def read_txt_to_list():
    res = []
    file = open('resume.txt', 'r')
    while 1:
        char = file.read(1)
        if not char:
            break
        res.append(char)
    file.close()
    return res


# 展示结果
def show_info(word_list, tag_list):
    word_res = []
    tag_res = []
    for i in range(len(word_list)):
        if not (tag_list[i].startswith('B') or tag_list[i].startswith('M') or tag_list[i].startswith('E')):
            word_res.append(word_list[i])
            tag_res.append('NULL')
            continue
        if tag_list[i].startswith('B'):
            str_word = word_list[i]
            str_tag = tag_list[i][2:]
            tag_res.append(str_tag)
            j = i + 1
            while 1:
                if tag_list[j].startswith('E'):
                    str_word += word_list[j]
                    break
                else:
                    str_word += word_list[j]
                    j += 1
            word_res.append(str_word)
    print("数据组合结果: ", word_res)
    print("数据标签: ", tag_res)
    return word_res, tag_res


# 修改格式
def format_resume():
    word_list = []
    tag_list = []
    with open('dr.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line[0]  # 内容（一个字）
            tag = line[2]  # 标签的开端
            i = 3
            while 1:
                if line[i] == '\t':  # 从标签的开端开始，直到'\t'，说明标签结束
                    break
                else:
                    tag += line[i]
                i += 1
            formatted_tag = ''
            if len(tag) > 1:
                formatted_tag = tag[0] + '-' + tag[5:].upper()
            else:
                formatted_tag = tag
            word_list.append(word)
            tag_list.append(formatted_tag)

    # 把tag list标签的最后一个改成E开头，中间的改成M开头
    if tag_list[-1].startswith('I'):
        tag_list[-1] = 'E' + tag_list[-1][1:]

    for i in range(len(tag_list) - 1):
        if tag_list[i].startswith('I'):
            if tag_list[i + 1].startswith('I') or tag_list[i + 1].startswith('E'):  # 后面那个以I或E开头，说明还在内部
                tag_list[i] = 'M' + tag_list[i][1:]
            else:  # 其他情况，说明这个标签结束了
                tag_list[i] = 'E' + tag_list[i][1:]
    return word_list, tag_list


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
