#! -*- coding: utf-8 -*-
# NEZHA model doing small talk tasks
# training script
# training environment：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.4

import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import os

# Use the second sheet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


maxlen = 256
batch_size = 16
steps_per_epoch = 10
epochs = 10

# nezha configuration
config_path = './PreTrainModel/nezha_gpt_dialog/config.json'
checkpoint_path = './PreTrainModel/nezha_gpt_dialog/model.ckpt'
dict_path = './PreTrainModel/nezha_gpt_dialog/vocab.txt'


def corpus():
    """loop reading corpus
    """
    while True:
        with open('LCCD-large-shuf.json') as f:
            for l in f:
                l = json.loads(l)
                yield l


# Load and refine the vocabulary
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

# Supplementary vocabulary
compound_tokens = []
for l in open('user_tokens.csv', encoding='utf-8'):
    token, count = l.strip().split('\t')
    if int(count) >= 10 and token not in token_dict:
        token_dict[token] = len(token_dict)
        compound_tokens.append([0])

# build tokenizer
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """data generator
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, texts in self.sample(random):
            token_ids, segment_ids = [tokenizer._token_start_id], [0]
            for i, text in enumerate(texts):
                ids = tokenizer.encode(text)[0][1:]
                if len(token_ids) + len(ids) <= maxlen:
                    token_ids.extend(ids)
                    segment_ids.extend([i % 2] * len(ids))
                else:
                    break
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """Cross entropy as loss, and mask off the padding part
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def mlm_acc(inputs):
        """The function to calculate the accuracy rate needs to be encapsulated as a layer
        """
        y_true, y_pred, mask = inputs
        y_true = y_true[:, 1:]  # Target token_ids
        y_pred = y_pred[:, :-1]  # Prediction sequence, staggered by one bit
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
    keep_tokens=keep_tokens,  # Only keep the words in keep_tokens, simplify the original word list
    compound_tokens=compound_tokens,  # vocabulary to expand
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.summary()

AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, 'AdamWG')
optimizer = AdamWG(
    learning_rate=2e-5,
    weight_decay_rate=0.01,
    exclude_from_weight_decay=['Norm', 'bias'],
    grad_accum_steps=16
)
model.compile(optimizer=optimizer)


class ChatBot(AutoRegressiveDecoder):
    """Chatbots based on random sampling
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    """save model weights
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
         while True:
             try:
                model.save_weights('./latest_model.weights')
                break
             except:
                 print(u'Save failed, retrying...')

        # val_acc = evaluate(valid_generator)
        # if val_acc > self.best_val_acc:
        #     self.best_val_acc = val_acc
        #     model.save_weights('best_model_tnews.weights')
        # print(
        #     u'val_acc: %.5f, best_val_acc: %.5f\n' %
        #     (val_acc, self.best_val_acc)
        # )



if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size)
    # record log
    csv_logger = keras.callbacks.CSVLogger('training_256.log')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator,csv_logger]
    )

else:

    model.load_weights('./latest_model.weights')
