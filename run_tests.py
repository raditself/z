
import unittest
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from tests.unit.test_game import TestGame
from tests.unit.test_model import TestModel
from tests.unit.test_mcts import TestMCTS
from tests.integration.test_self_play import TestSelfPlay

def run_tests():
    # Create a test suite combining all test cases
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestGame))
    test_suite.addTest(unittest.makeSuite(TestModel))
    test_suite.addTest(unittest.makeSuite(TestMCTS))
    test_suite.addTest(unittest.makeSuite(TestSelfPlay))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
