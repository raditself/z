
import unittest
import torch
import os
from src.alphazero.model import AlphaZeroNetwork

def export_model(model, path):
    torch.save(model.state_dict(), path)

def import_model(model, path):
    model.load_state_dict(torch.load(path))

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (3, 8, 8)  # Example input shape for 8x8 chess board with 3 input planes
        self.action_size = 4672  # Example action size for chess
        self.model = AlphaZeroNetwork(self.input_shape, self.action_size)

    def test_forward_pass(self):
        batch_size = 1
        x = torch.randn(batch_size, *self.input_shape)
        game_phase = torch.rand(batch_size, 1)  # Random game phase between 0 and 1
        policy, value = self.model(x, game_phase)
        
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

    def test_game_phase_impact(self):
        batch_size = 1
        x = torch.randn(batch_size, *self.input_shape)
        
        # Test with different game phases
        early_game = torch.tensor([[0.1]])
        mid_game = torch.tensor([[0.5]])
        late_game = torch.tensor([[0.9]])
        
        policy1, value1 = self.model(x, early_game)
        policy2, value2 = self.model(x, mid_game)
        policy3, value3 = self.model(x, late_game)
        
        # Check if the values are different for different game phases
        self.assertFalse(torch.allclose(value1, value2))
        self.assertFalse(torch.allclose(value2, value3))
        self.assertFalse(torch.allclose(value1, value3))
        
        # Check if policies are the same (should not be affected by game phase)
        self.assertTrue(torch.allclose(policy1, policy2))
        self.assertTrue(torch.allclose(policy2, policy3))

if __name__ == '__main__':
    unittest.main()
