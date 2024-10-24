
import unittest
import torch
from src.alphazero.model import AlphaZeroNetwork

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (3, 8, 8)  # Example input shape for 8x8 chess board with 3 input planes
        self.action_size = 4672  # Example action size for chess
        self.model = AlphaZeroNetwork(self.input_shape, self.action_size)

    def test_forward_pass(self):
        batch_size = 1
        x = torch.randn(batch_size, *self.input_shape)
        policy, value = self.model(x)
        
        self.assertEqual(policy.shape, (batch_size, self.action_size))
        self.assertEqual(value.shape, (batch_size, 1))

    def test_save_load(self):
        # Test export_model and import_model functions
        model_path = 'test_model.pth'
        export_model(self.model, model_path)
        
        new_model = AlphaZeroNetwork(self.input_shape, self.action_size)
        import_model(new_model, model_path)
        
        # Check if parameters are the same after save and load
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

if __name__ == '__main__':
    unittest.main()
