import os
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from src.model import RNNModel, RNNModelTrain
from src.utils import generate_batch, process_poems

def parse_args():
    """set and check parameters"""
    parser = argparse.ArgumentParser(description='train poems')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epoches for training. (default: %(default)d)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate used to train with warmup. (default: %(default)f)')
    parser.add_argument("--train_data_file_path", type=str, default="./data/poems.txt",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--save_checkpoint_path", type=str, default="./ckpt", help="Save checkpoint path")
    args = parser.parse_args()

    return args

def run_train(args):
    if not os.path.exists(args.save_checkpoint_path):
        os.makedirs(args.save_checkpoint_path)
    
    poems_vector, word_to_int, vocabularies = process_poems(args.train_data_file_path)
    batches_inputs, batches_outputs = generate_batch(args.batch_size, poems_vector, word_to_int)
    
    net = RNNModel(len(vocabularies), rnn_size=128, model='lstm', padding_idx=word_to_int[' '])
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_loss = RNNModelTrain(net, len(vocabularies))
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.learning_rate)
    trainer = nn.TrainOneStepCell(net_with_loss, optimizer)

    n_chunk = len(poems_vector) // args.batch_size
    for epoch in range(args.epochs):
        for batch in range(n_chunk):
            x_batch = Tensor(batches_inputs[batch], mindspore.int32)
            y_batch = Tensor(batches_outputs[batch], mindspore.int32)
            # print(x_batch.shape, y_batch.shape, seq_length.shape)
            loss = trainer(x_batch, y_batch.reshape(-1))
            print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss.asnumpy()))
        if epoch % 6 == 0:
            mindspore.save_checkpoint(net, os.path.join(args.save_checkpoint_path, f'poems.{epoch}.ckpt'))

if __name__ == '__main__':
    args = parse_args()
    run_train(args)