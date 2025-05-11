import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def split_and_encode(df, label_col='airline_sentiment'):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_df[label_col])
    test_labels = le.transform(test_df[label_col])

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    return train_df, test_df, train_labels, test_labels