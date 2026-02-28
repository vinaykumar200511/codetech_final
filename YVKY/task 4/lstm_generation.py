# ================================
# LSTM Text Generation - Complete Code
# ================================

import numpy as np
import tensorflow as tf
LSTM = tf.keras.layers.LSTM
Embedding = tf.keras.layers.Embedding

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
to_categorical = tf.keras.utils.to_categorical

# --------------------------------
# 1. Sample Training Dataset
# --------------------------------
texts = [
    "Artificial intelligence is transforming industries",
    "Machine learning enables predictive analytics",
    "Deep learning powers modern AI systems",
    "Neural networks learn from data",
    "AI is the future of technology"
]

# --------------------------------
# 2. Tokenization
# --------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

total_words = len(tokenizer.word_index) + 1

# --------------------------------
# 3. Create Input Sequences
# --------------------------------
input_sequences = []

for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert y to categorical
y = to_categorical(y, num_classes=total_words)

# --------------------------------
# 4. Build LSTM Model
# --------------------------------
model = Sequential([
    Embedding(total_words, 64, input_length=max_sequence_len - 1),
    LSTM(128),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# --------------------------------
# 5. Train Model
# --------------------------------
print("Training model...")
model.fit(X, y, epochs=200, verbose=1)

print("✅ Training Completed!")

# --------------------------------
# 6. Text Generation Function
# --------------------------------
def generate_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# --------------------------------
# 7. Generate Sample Text
# --------------------------------
print("\nGenerated Text:")
print(generate_text("Artificial", 8))