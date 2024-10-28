
import numpy as np
import torch
from src.alphazero.mcts import AdaptiveMCTS
from src.alphazero.neural_network import DynamicNeuralNetwork

class MultiAgentAlphaZero:
    def __init__(self, game, num_agents):
        self.game = game
        self.num_agents = num_agents
        self.networks = [DynamicNeuralNetwork(game) for _ in range(num_agents)]
        self.mcts_agents = [AdaptiveMCTS(game, network) for network in self.networks]

    def self_play(self, num_games=100, num_simulations=800):
        game_data = [[] for _ in range(self.num_agents)]

        for _ in range(num_games):
            state = self.game.get_initial_state()
            current_player = 0
            game_history = []

            while not self.game.is_terminal(state):
                mcts = self.mcts_agents[current_player]
                action_probs = mcts.get_action_prob(state, num_simulations)
                action = np.random.choice(len(action_probs), p=action_probs)
                
                game_history.append((state, action_probs, current_player))
                state = self.game.get_next_state(state, action)
                current_player = (current_player + 1) % self.num_agents

            rewards = self.game.get_rewards(state)

            for hist_state, hist_probs, hist_player in game_history:
                game_data[hist_player].append((
                    hist_state,
                    hist_probs,
                    rewards[hist_player]
                ))

        return game_data

    def train(self, game_data, num_epochs=10, batch_size=32):
        for agent_id, agent_data in enumerate(game_data):
            network = self.networks[agent_id]
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

            for _ in range(num_epochs):
                np.random.shuffle(agent_data)
                for batch_start in range(0, len(agent_data), batch_size):
                    batch = agent_data[batch_start:batch_start+batch_size]
                    states, pis, vs = zip(*batch)

                    states = torch.FloatTensor(np.array(states))
                    pis = torch.FloatTensor(np.array(pis))
                    vs = torch.FloatTensor(np.array(vs))

                    out_pis, out_vs = network(states)

                    pi_loss = -torch.sum(pis * torch.log(out_pis + 1e-8)) / pis.size()[0]
                    v_loss = torch.sum((vs - out_vs.view(-1)) ** 2) / vs.size()[0]
                    total_loss = pi_loss + v_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    def update_mcts(self):
        for agent_id, network in enumerate(self.networks):
            self.mcts_agents[agent_id].update_network(network)

if __name__ == "__main__":
    from src.games.multiplayer_game import MultiplayerGame

    game = MultiplayerGame(num_players=3)
    multi_agent_alphazero = MultiAgentAlphaZero(game, num_agents=3)

    for iteration in range(50):
        print(f"Iteration {iteration + 1}")
        game_data = multi_agent_alphazero.self_play()
        multi_agent_alphazero.train(game_data)
        multi_agent_alphazero.update_mcts()

    print("Multi-agent AlphaZero training complete!")
