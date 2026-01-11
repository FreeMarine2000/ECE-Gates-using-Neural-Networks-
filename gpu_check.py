import tensorflow as tf
import sys

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\nFound GPU: {gpus[0]}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
else:
    print("\n No GPU found.")