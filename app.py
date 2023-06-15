from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = Flask(__name__)

model = tf.keras.models.load_model("mymodel.h5")
with open('tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
with open('label_encoder.json') as f:
    label_encoder_json = f.read()
    label_encoder = json.loads(label_encoder_json)

# Convert label_encoder list to dictionary
label_encoder = {str(i): label for i, label in enumerate(label_encoder)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['description']
    preprocessed_input = preprocess_input(data, tokenizer, model.input_shape[1])
    predicted_probabilities = model.predict(preprocessed_input)
    predicted_class_index = tf.argmax(predicted_probabilities, axis=1).numpy()[0]
    predicted_class = label_encoder[str(predicted_class_index)]
    return json.dumps({'predicted_class': predicted_class})

def preprocess_input(input_text, tokenizer, max_length):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    return padded_sequence

if __name__ == '__main__':
    app.run(debug=True)
