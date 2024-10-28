
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

class StepDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        return self.initial_learning_rate * self.decay_rate ** (step // self.decay_steps)

class AlphaZeroModel:
    def __init__(self, input_shape=(8, 8, 14), num_actions=4672, initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        game_phase_input = layers.Input(shape=(1,))
        
        # Initial convolutional layer
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Residual blocks
        for _ in range(19):  # AlphaZero uses 19 or 39 residual blocks
            x = self.residual_block(x)

        # Multi-headed policy
        policy_head_opening = self.policy_head(x, 'opening')
        policy_head_middlegame = self.policy_head(x, 'middlegame')
        policy_head_endgame = self.policy_head(x, 'endgame')

        # Combine policy heads based on game phase
        policy_output = self.combine_policy_heads(policy_head_opening, policy_head_middlegame, policy_head_endgame, game_phase_input)

        # Value head
        value_head = layers.Conv2D(1, 1, use_bias=False)(x)
        value_head = layers.BatchNormalization()(value_head)
        value_head = layers.ReLU()(value_head)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(256, activation='relu')(value_head)
        value_output = layers.Dense(1, activation='tanh', name='value')(value_head)

        model = models.Model(inputs=[inputs, game_phase_input], outputs=[policy_output, value_output])
        
        lr_schedule = StepDecaySchedule(self.initial_learning_rate, self.decay_steps, self.decay_rate)
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(optimizer=optimizer,
                      loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      loss_weights={'policy': 1.0, 'value': 1.0})
        
        return model

    def residual_block(self, x):
        residual = x
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([residual, x])
        x = layers.ReLU()(x)
        return x

    def policy_head(self, x, name):
        policy_head = layers.Conv2D(2, 1, use_bias=False)(x)
        policy_head = layers.BatchNormalization()(policy_head)
        policy_head = layers.ReLU()(policy_head)
        policy_head = layers.Flatten()(policy_head)
        return layers.Dense(self.num_actions, activation='softmax', name=f'policy_{name}')(policy_head)

    def combine_policy_heads(self, opening, middlegame, endgame, game_phase):
        opening_weight = layers.Lambda(lambda x: tf.maximum(0.0, 1.0 - x * 3))(game_phase)
        middlegame_weight = layers.Lambda(lambda x: tf.maximum(0.0, tf.minimum(1.0, 3.0 * x) - tf.maximum(0.0, 3.0 * x - 2.0)))(game_phase)
        endgame_weight = layers.Lambda(lambda x: tf.maximum(0.0, 3.0 * x - 2.0))(game_phase)
        
        combined_policy = layers.Add()([
            layers.Multiply()([opening, opening_weight]),
            layers.Multiply()([middlegame, middlegame_weight]),
            layers.Multiply()([endgame, endgame_weight])
        ])
        
        return layers.Lambda(lambda x: x / tf.reduce_sum(x, axis=-1, keepdims=True), name='policy')(combined_policy)

    def predict(self, state, game_phase):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        game_phase = np.array([[game_phase]])  # Convert to 2D array
        policy, value = self.model.predict([state, game_phase])
        return policy[0], value[0][0]  # Remove batch dimension
