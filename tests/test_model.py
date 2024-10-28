
import unittest
import numpy as np
from src.alphazero.model import ChessModel

class TestChessModel(unittest.TestCase):
    def setUp(self):
        self.model = ChessModel()

    def test_model_structure(self):
        self.assertIsNotNone(self.model.model)
        self.assertEqual(len(self.model.model.layers), 5)  # Input, 2 hidden, policy output, value output

    def test_predict(self):
        # Create a dummy input (8x8x12 board representation)
        dummy_input = np.random.rand(1, 8, 8, 12)
        policy, value = self.model.predict(dummy_input)

        self.assertEqual(policy.shape, (1, 64))  # 64 possible moves in chess
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(np.all(policy >= 0) and np.all(policy <= 1))
        self.assertTrue(-1 <= value <= 1)

    def test_train(self):
        # Create dummy training data
        states = np.random.rand(10, 8, 8, 12)
        policy_targets = np.random.rand(10, 64)
        policy_targets /= np.sum(policy_targets, axis=1, keepdims=True)
        value_targets = np.random.uniform(-1, 1, (10, 1))

        # Train the model
        loss = self.model.train(states, policy_targets, value_targets)
        self.assertIsInstance(loss, float)

    def test_save_and_load(self):
        # Save the model
        self.model.save('test_model.h5')

        # Create a new model and load the saved weights
        new_model = ChessModel()
        new_model.load('test_model.h5')

        # Check if the weights are the same
        for layer1, layer2 in zip(self.model.model.layers, new_model.model.layers):
            if layer1.get_weights():
                self.assertTrue(np.all(layer1.get_weights()[0] == layer2.get_weights()[0]))

        # Clean up
        import os
        os.remove('test_model.h5')

if __name__ == '__main__':
    unittest.main()
