#! -*- coding: utf-8 -*-
# NEZHA model doing small talk tasks
# test script

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder

# nezha configuration
config_path = './PreTrainModel/nezha_gpt_dialog/config.json'
checkpoint_path = './PreTrainModel/nezha_gpt_dialog/model.ckpt'
dict_path = './PreTrainModel/nezha_gpt_dialog/vocab.txt'



# build tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# Build and load the model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
)
model.summary()


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

# print(chatbot.response([u'Don't love me with no results', u'you will lose me like this', u'what if you lose']))


history = []
while True:
    raw_text = input(">>> ")
    while not raw_text:
        print('Input can not be empty!')
        raw_text = input(">>> ")
    raw_text = " ".join(raw_text)
    history.append(raw_text)

    out_text = chatbot.response(history)
    print(out_text)


