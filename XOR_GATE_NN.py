import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(f"TensorFlow Version: {tf.__version__}")
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("running it 500 x")
model.fit(X, y, epochs=500, verbose=0) 
print("Training complete.")
print("predict")
predictions = model.predict(X)
for input_val, pred in zip(X, predictions):
    predicted_label = 1 if pred[0] > 0.5 else 0
    print(f"Input: {input_val} | Raw Output: {pred[0]:.4f} | Result: {predicted_label}")