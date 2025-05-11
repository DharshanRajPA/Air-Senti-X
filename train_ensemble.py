import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from preprocessing.data_cleaning import preprocess_dataset
from preprocessing.data_split_encode import split_and_encode
from preprocessing.tokenize_bert import bert_tokenize

from models.architectures.bert_lstm import build_bert_lstm
from models.architectures.bert_bilstm import build_bert_bilstm
from models.architectures.bert_cnn import build_bert_cnn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename='training.log',
    filemode='w'
)
logger = logging.getLogger()

log_dir = os.path.join("logs", "fit")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def get_callbacks(model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("models", "saved", f"{model_name}.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
    )
    return [checkpoint, early_stop, tensorboard_callback]

DATA_PATH = 'dataset/Tweets.csv'   
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
ensemble_weights = [1.0, 1.0, 1.0]
CONFIDENCE_THRESHOLD = 0.5 

logger.info("Cleaning dataset...")
print("[INFO] Cleaning dataset...")
df = preprocess_dataset(DATA_PATH)

logger.info("Splitting dataset and encoding labels...")
print("[INFO] Splitting and encoding labels...")
train_df, test_df, train_labels, test_labels = split_and_encode(df, label_col='airline_sentiment')

logger.info("Tokenizing texts using BERT tokenizer...")
print("[INFO] Tokenizing text using BERT...")
X_train_input_ids, X_train_attention_masks = bert_tokenize(train_df['text'].tolist(), max_len=MAX_LEN)
X_test_input_ids, X_test_attention_masks = bert_tokenize(test_df['text'].tolist(), max_len=MAX_LEN)

num_labels = len(set(train_labels))
logger.info(f"Number of classes: {num_labels}")
print(f"[INFO] Number of classes: {num_labels}")

logger.info("Building models...")
print("[INFO] Building models...")
model_lstm = build_bert_lstm(MAX_LEN, num_labels)
model_bilstm = build_bert_bilstm(MAX_LEN, num_labels)
model_cnn = build_bert_cnn(MAX_LEN, num_labels)

model_names = ['bert_lstm', 'bert_bilstm', 'bert_cnn']
models = [model_lstm, model_bilstm, model_cnn]

for model in models:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def train_model(model, model_name):
    logger.info(f"Training {model_name}...")
    print(f"[INFO] Training {model_name} model...")
    callbacks = get_callbacks(model_name)
    model.fit(
        [X_train_input_ids, X_train_attention_masks],
        train_labels,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    logger.info(f"{model_name} training complete!")
    print(f"[INFO] {model_name} training complete!")
    preds = model.predict([X_test_input_ids, X_test_attention_masks])
    return preds

preds_list = []
for model, name in zip(models, model_names):
    preds = train_model(model, name)
    preds_list.append(preds)

logger.info("Ensembling predictions...")
print("[INFO] Averaging predictions for ensemble...")
preds_array = np.array(preds_list)  

weighted_preds = np.average(preds_array, axis=0, weights=ensemble_weights)
final_preds = tf.argmax(weighted_preds, axis=1).numpy()

max_confidences = np.max(weighted_preds, axis=1)

low_confidence_count = np.sum(max_confidences < CONFIDENCE_THRESHOLD)
logger.info(f"Found {low_confidence_count} low-confidence predictions (threshold: {CONFIDENCE_THRESHOLD}).")
print(f"[INFO] {low_confidence_count} predictions below confidence threshold.")

logger.info("Evaluating ensemble performance...")
print("[RESULT] Ensemble Classification Report:")
report = classification_report(test_labels, final_preds)
conf_matrix = confusion_matrix(test_labels, final_preds)
print(report)
print("Confusion Matrix:")
print(conf_matrix)
logger.info("Classification Report:\n" + report)
logger.info("Confusion Matrix:\n" + str(conf_matrix))

for model, name in zip(models, model_names):
    save_path = os.path.join("models", "saved", f"{name}.h5")
    model.save(save_path)
    logger.info(f"{name} model saved at {save_path}")
    print(f"[INFO] {name} model saved at {save_path}")

try:
    import tf2onnx
    best_model_index = np.argmax([np.mean(np.max(pred, axis=1)) for pred in preds_list])
    best_model = models[best_model_index]
    onnx_path = os.path.join("models", "saved", f"{model_names[best_model_index]}.onnx")
    spec = (tf.TensorSpec(best_model.inputs[0].shape, tf.int32, name="input_ids"),
            tf.TensorSpec(best_model.inputs[1].shape, tf.int32, name="attention_mask"))
    model_proto, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=13)
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    logger.info(f"Best model converted to ONNX and saved at {onnx_path}")
    print(f"[INFO] Best model converted to ONNX and saved at {onnx_path}")
except ImportError as e:
    logger.warning("tf2onnx not installed. Skipping ONNX conversion.")
    print("[WARNING] tf2onnx not installed. Skipping ONNX conversion.")
except Exception as e:
    logger.error("Error converting model to ONNX: " + str(e))
    print("[ERROR] Error converting model to ONNX:", e)

logger.info("Training and ensemble evaluation complete.")
print("[INFO] Training and ensemble evaluation complete.")
