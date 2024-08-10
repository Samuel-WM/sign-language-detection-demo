import os
import tensorflow as tf

tf_path = os.path.dirname(tf.__file__)
init_file = os.path.join(tf_path, '__init__.py')

print(f'TensorFlow __init__.py is located at: {init_file}')
