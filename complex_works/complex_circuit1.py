import tensorflow as tf
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(precision=3, suppress=True)

print("Generating Data...")
X = []
y = []
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            for d in [0,1]:
                X.append([a, b, c, d])
                y.append([(a and b) ^ (c and d)])
X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    
    tf.keras.layers.Dense(16, activation='tanh', name='hidden_layer'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Training (2000 epochs)...")
model.fit(X, y, epochs=2000, verbose=0)
print("Training Complete.\n")
test_input = np.array([[1, 1, 0, 0]], dtype="float32")
prediction = model.predict(test_input, verbose=0)
print(f"--- RESULTS FOR INPUT {test_input[0]} ---")
print(f"Final Output Probability: {prediction[0][0]:.4f}")
print(f"Logic Result: {int(round(prediction[0][0]))}")
hidden_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('hidden_layer').output)
hidden_activations = hidden_model.predict(test_input, verbose=0)
print("\n--- INTERNAL: HIDDEN LAYER ACTIVATIONS ---")
print("What the 16 neurons are outputting (-1.0 to 1.0):")
print(hidden_activations[0])
weights, biases = model.get_layer('hidden_layer').get_weights()
print("\n--- INTERNAL: WIRING SAMPLE ---")
print("How strictly Neuron #0 judges the 4 inputs:")
print(f"Weights: {weights[:, 0]}") 
print("(Positive = Excites the neuron, Negative = Inhibits it)")