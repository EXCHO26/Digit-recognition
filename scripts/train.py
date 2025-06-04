from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from utils.data_loader import load_data # type: ignore

from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore

def build_model():
    """Define the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    # Load data
    (X_train, y_train), (X_val, y_val) = load_data()
    
    # Build and train model
    model = build_model()
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('models/mnist_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

if __name__ == "__main__":
    train()