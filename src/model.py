import mindspore.nn as nn
from mindspore.common.initializer import Uniform, TruncatedNormal

class RNNModel(nn.Cell):
    def __init__(self, vocab_size, rnn_size=128, num_layers=2, model:str='rnn', padding_idx=0):
        super().__init__()
        self.rnn_size = rnn_size
        if model == 'rnn':
            self.rnn = nn.RNN(rnn_size, rnn_size, num_layers, batch_first=True)
        elif model == 'gru':
            self.rnn = nn.GRU(rnn_size, rnn_size, num_layers, batch_first=True)
        elif model == 'lstm':
            self.rnn = nn.LSTM(rnn_size, rnn_size, num_layers, batch_first=True)
        else:
            raise ValueError('Not supported model: {}'.format(model))
        
        self.embedding = nn.Embedding(vocab_size + 1, rnn_size, embedding_table=Uniform(1.0), padding_idx=padding_idx)
        self.fc = nn.Dense(rnn_size, vocab_size + 1, weight_init=TruncatedNormal(1.0))

    def construct(self, inputs, seq_length=None):
        embed_inputs = self.embedding(inputs)
        outputs, _ = self.rnn(embed_inputs, seq_length=seq_length)
        outputs = outputs.reshape((-1, self.rnn_size))
        logits = self.fc(outputs)
        return logits

class RNNModelTrain(nn.Cell):
    def __init__(self, net, vocab_size):
        super().__init__()
        self.net = net
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        self.onehot = nn.OneHot(depth=vocab_size+1)

    def construct(self, inputs, labels):
        labels = self.onehot(labels)
        logits = self.net(inputs)
        return self.loss(logits, labels)

class RNNModelInfer(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.softmax = nn.Softmax()
    
    def construct(self, inputs):
        logits = self.net(inputs)
        return self.softmax(logits)