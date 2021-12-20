import os
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from src.model import RNNModel, RNNModelInfer
from src.utils import process_poems

start_token = 'B'
end_token = 'E'
model_dir = './ckpt/'
corpus_file = './data/poems.txt'

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(begin_word):
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)
    print(len(vocabularies))
    rnn_model = RNNModel(len(vocabularies), rnn_size=128, model='lstm')
    param_dict = load_checkpoint(
        os.path.join(model_dir, f'poems.6.ckpt'))
    param_not_load = load_param_into_net(rnn_model, param_dict)
    print(param_not_load)
    rnn_model = RNNModelInfer(rnn_model)
    x = np.array([list(map(word_int_map.get, start_token))])
    predict = rnn_model(Tensor(x, mindspore.int32))
    
    word = begin_word or to_word(predict.asnumpy(), vocabularies)
    poem_ = ''

    i = 0
    while word != end_token:
        poem_ += word
        i += 1
        if i > 24:
            break
        x = np.array([[word_int_map[word]]])
        predict = rnn_model(Tensor(x, mindspore.int32))
        word = to_word(predict.asnumpy(), vocabularies)

    return poem_

if __name__ == '__main__':
    begin_char = input('## （输入 quit 退出）请输入第一个字 please input the first character: ')
    if begin_char == 'quit':
        exit() 
    poem = gen_poem(begin_char)
    print(poem)