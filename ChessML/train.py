import tensorflow as tf
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers

# Run: model = build_model(32, 4) # Convolutional size and depth
def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(conv_size, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    
    return models.Model(inputs=board3d, outputs=x)

# Residual Network
def build_model_residual(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding="same", activation="relu")(board3d)
    
    for _ in range(conv_depth):
        previous_x = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous_x])
        x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    
    return models.Model(inputs=board3d, outputs=x)

def main():
    model = build_model(32, 4)
    model.compile(optimizer=optimizers.Adam(5e-4), loss="mean_squared_error", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, batch_size=2048, epochs=1000, verbose=1, validation_split=0.1, callbacks=[tensorboard_callback])