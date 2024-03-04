import tensorflow as tf
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.callbacks import TensorBoard
import time
import data_parser


# Run: model = build_model(32, 4) # Convolutional size and depth
class EvaluationModel:
    def __init__(self, layer_count, batch_size, learning_rate=1e-3):
        self.layer_count = layer_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        for _ in range(self.layer_count - 1):
            model.add(layers.Dense(808, activation="relu"))
        model.add(layers.Dense(1))
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_absolute_error",
        )

        model.summary()
        return model

    def train(self, x_train, y_train, epochs):
        version_name = f"{int(time.time())}-batch_size-{self.batch_size}-layer_count-{self.layer_count}"
        tensorboard_callback = TensorBoard(log_dir=f"./logs/{version_name}")
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[tensorboard_callback],
        )


def main():
    model = EvaluationModel(
        layer_count=10,
        batch_size=1024,
        learning_rate=1e-3,
    )
    model.create_model()

    data_base = data_parser.EvaluationDataset()
    data_base.connect()
    data_base.import_game(".\\ChessML\\Dataset\\lichess_db_standard_rated_2024-02.pgn")

    for i in range(10):
        data = next(data_base)
        x_train = data["binary"]
        y_train = data["eval"]
        model.train(x_train, y_train, epochs=10)

    model.save(".\\ChessML\\checkpoints\\model.h5")

if __name__ == "__main__":
    main()