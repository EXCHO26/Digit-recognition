from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def load_data():
    """Load and preprocess MNIST data."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    return (X_train, y_train), (X_test, y_test)