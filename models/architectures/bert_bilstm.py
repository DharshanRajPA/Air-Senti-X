import tensorflow as tf
from transformers import TFBertModel
from keras.layers import Input, Bidirectional, LSTM, Dense
from keras.models import Model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

def build_bert_bilstm(max_len, num_labels):
    input_ids = Input(shape=(max_len,), dtype='int32', name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype='int32', name="attention_mask")

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_model.trainable = True

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    x = Bidirectional(LSTM(128))(bert_output)
    output = Dense(num_labels, activation='softmax')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    return model
