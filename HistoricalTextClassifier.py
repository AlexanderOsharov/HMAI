import chardet
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder

class HistoricalTextClassifier:
    def __init__(self, dataset_path='dataset.json'):
        self.dataset_path = dataset_path
        self.tokenizer = None
        self.model = None
        self.sequences = []
        self.labels = []
        self.max_sequence_length = 0

        self.load_dataset()  # Загрузка данных из JSON файла
        self.load_or_create_model()  # Загрузка существующей модели или ее создание

    def load_dataset(self):
        # Определяем кодировку файла
        with open(self.dataset_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Читаем файл с использованием определенной кодировки
        with open(self.dataset_path, 'r', encoding=encoding, errors='replace') as file:
            data = json.load(file)

        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.sequences = pad_sequences(sequences, padding='post')
        self.labels = encoded_labels
        self.max_sequence_length = max(len(seq) for seq in sequences)

    def load_or_create_model(self):
        try:
            self.model = load_model('historical_text_classifier.h5')
        except FileNotFoundError:
            self.create_model()

    def create_model(self):
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128,
                      input_length=self.max_sequence_length),
            LSTM(128, return_sequences=True),
            Dropout(0.5),
            LSTM(64),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.train_model()
        self.model.save('historical_text_classifier.h5')

    def train_model(self):
        self.model.fit(self.sequences, self.labels, epochs=30, batch_size=32)

    def predict(self, text):
        new_sequence = self.tokenizer.texts_to_sequences([text])
        padded_new_sequence = pad_sequences(new_sequence, maxlen=self.max_sequence_length)
        prediction = self.model.predict(padded_new_sequence)[0][0]
        return "historical_background" if prediction > 0.5 else "twaddle"

    def update_dataset(self, text, label):
        new_sequence = self.tokenizer.texts_to_sequences([text])[0]
        self.sequences = np.append(self.sequences, [new_sequence], axis=0)
        self.labels = np.append(self.labels, label)
        self.sequences = pad_sequences(self.sequences, maxlen=self.max_sequence_length)
        self.labels = np.array(self.labels)
        self.train_model()