import tensorflow as tf
import numpy as np
import os 
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(f"TensorFlow Version: {tf.__version__}")
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype="float32")
y = np.array([[0], [1], [1], [0]], dtype="float32")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(8, input_dim=2, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("running it nfinite x")
start_time = time.time()
model.fit(X, y, epochs=1000, verbose=0) 
end_time = time.time()
print(f"Training time: {end_time - start_time:.4f} seconds")
print("Training complete.")
print("predict")
predictions = model.predict(X)
for input_val, pred in zip(X, predictions):
    predicted_label = 1 if pred[0] > 0.5 else 0
    print(f"Input: {input_val} | Raw Output: {pred[0]:.4f} | Result: {predicted_label}")