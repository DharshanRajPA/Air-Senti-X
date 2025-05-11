from transformers import BertTokenizer
import tensorflow as tf
import pickle

MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

def bert_tokenize(texts, max_len=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return (
        tf.concat(input_ids, axis=0),
        tf.concat(attention_masks, axis=0)
    )
