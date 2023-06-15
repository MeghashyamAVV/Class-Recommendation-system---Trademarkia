import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential

def solution_model():
    vocab_size = 10000
    embedding_dim = 30
    max_length = 40
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 50000
    num_classes = 45  # Update the number of classes

    sentences = []
    labels = []

    with open("shuffled_file.json", 'r') as f:
        datastore = json.load(f)

    for item in datastore:
        sentences.append(item['description'])
        labels.append(item['class_id'])

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='mymodel.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    label_encoder = LabelEncoder()
    training_labels_encoded = label_encoder.fit_transform(training_labels)
    testing_labels_encoded = label_encoder.transform(testing_labels)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        training_padded,
        training_labels_encoded,
        epochs=num_epochs,
        validation_data=(testing_padded, testing_labels_encoded),
        batch_size=256,
        verbose=1,
        callbacks=callbacks
    )

    # Save the tokenizer as a JSON file
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_json)

    # Save the label encoder as a JSON file
    label_encoder_json = json.dumps(list(label_encoder.classes_))
    with open('label_encoder.json', 'w') as f:
        f.write(label_encoder_json)

    return model

if __name__ == '__main__':
    num_epochs = 30
    model = solution_model()
    model.save("mymodel.h5")
