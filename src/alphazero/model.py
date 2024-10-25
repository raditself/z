import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        residual = x
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return tf.nn.relu(out)

class ChessModel(tf.keras.Model):
    def __init__(self, input_shape=(8, 8, 14), num_filters=256, num_res_blocks=19):
        super(ChessModel, self).__init__()
        self.input_shape = input_shape
        self.conv = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', input_shape=input_shape)
        self.bn = tf.keras.layers.BatchNormalization()
        self.res_blocks = [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        self.policy_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4672)  # 4672 is the number of possible moves in chess
        ])
        self.value_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])

    def call(self, x):
        x = tf.nn.relu(self.bn(self.conv(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def get_model(model_path=None):
    model = ChessModel()
    if model_path:
        model.load_weights(model_path)
    return model

def get_data_loader(data, batch_size=32, shuffle=True):
    def generator():
        for board, policy, value in data:
            yield (tf.convert_to_tensor(board, dtype=tf.float32),
                   tf.convert_to_tensor(policy, dtype=tf.float32),
                   tf.convert_to_tensor([value], dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(8, 8, 14), dtype=tf.float32),
            tf.TensorSpec(shape=(4672,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

