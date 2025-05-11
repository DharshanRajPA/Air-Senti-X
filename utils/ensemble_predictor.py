import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from keras.models import load_model
from scipy.special import softmax
from transformers import TFBertModel


class EnsemblePredictor:
    def __init__(self, model_paths, tokenizer_path=None, label_encoder_path=None,
                 tokenizer_name='bert-base-uncased', max_len=128):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[INFO] Using GPU")
            except RuntimeError as e:
                print(e)

        self.models = [
            load_model(path, compile=False, custom_objects={"TFBertModel": TFBertModel})
            for path in model_paths
        ]
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.labels = ['Negative', 'Neutral', 'Positive']

    def preprocess(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='np',
            return_attention_mask=True
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }

    def predict(self, text):
        inputs = self.preprocess(text)
        predictions = []

        for model in self.models:
            pred = model.predict(inputs, verbose=0)
            predictions.append(pred)

        avg_pred = np.mean(predictions, axis=0)
        probs = softmax(avg_pred[0])  # Softmax for probability
        label_index = int(np.argmax(probs))
        confidence = float(np.max(probs))

        result = {
            "label": self.labels[label_index],
            "confidence": round(confidence, 4),
            "probabilities": {label: round(float(prob), 4) for label, prob in zip(self.labels, probs)}
        }

        return result
