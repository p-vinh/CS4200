import tensorflow as tf
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import time
import data_parser


# Run: model = build_model(32, 4) # Convolutional size and depth
class EvaluationModel:
    def __init__(self, layer_count, batch_size, learning_rate=1e-3):
        self.layer_count = layer_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def create_model(self, conv_size, conv_depth):
        board3d = layers.Input(shape=(14, 8, 8))
        
        x = board3d
        
        for _ in range(conv_depth):
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=board3d, outputs=x)
    
    def create_model_residual(self, conv_size, conv_depth):
        board3d = layers.Input(shape=(14, 8, 8))
        
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last', activation='relu')(x)
        
        for _ in range(conv_depth):
            previous = x
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_last', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, previous])
            x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=board3d, outputs=x)

    def create_model_rec(self):
        model = models.Sequential()

        model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=(14, 8, 8), activation='relu'))
        model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
        model.add(layers.Conv2D(32, (1, 1), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (1, 1), padding='same', activation='relu'))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return models.Model(inputs=board3d, outputs=x)

    def train(self, x_train, y_train, epochs):
        version_name = f"{int(time.time())}-batch_size-{self.batch_size}-layer_count-{self.layer_count}"
        tensorboard_callback = TensorBoard(log_dir=f"./logs/{version_name}")
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        self.model.summary()
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_split=0.1,
            callbacks=[ReduceLROnPlateau(moniter='loss', patience=10),
                       EarlyStopping(monitor='loss', patience=15, min_delta=0.001),
                       tensorboard_callback],
        )
        
        self.model.save(f"./checkpoints/{version_name}.h5")


def main():
    model = EvaluationModel(
        layer_count=10,
        batch_size=1024,
        learning_rate=5e-4
    )
    
    model_chess = model.create_model(32, 4)
    
    data_base = data_parser.EvaluationDataset()
    # data_base.import_game(".\\ChessML\\Dataset\\lichess_db_standard_rated_2024-02.pgn")

    while True:
        try:
            data = next(data_base)
            if data is not None:
                x_train = data["binary"]
                y_train = data["eval"]
                model_chess.train(x_train, y_train, epochs=1000)
        except StopIteration:
            break
    data_base.close()
    
if __name__ == "__main__":
    main()