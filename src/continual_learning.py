
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ContinualLearningMechanism:
    def __init__(self, model, learning_rate=0.001, ewc_lambda=0.4):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.optimal_params = {}

    def calculate_fisher_information(self, data_loader):
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for batch in data_loader:
            self.model.zero_grad()
            output = self.model(batch)
            loss = nn.functional.nll_loss(output, batch.target)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                fisher_info[n] += p.grad.data ** 2 / len(data_loader)
        
        self.fisher_information = fisher_info
        self.optimal_params = {n: p.clone() for n, p in self.model.named_parameters()}

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            _loss = self.fisher_information[n] * (p - self.optimal_params[n]) ** 2
            loss += _loss.sum()
        return self.ewc_lambda * loss

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(batch)
        task_loss = nn.functional.nll_loss(output, batch.target)
        ewc_loss = self.ewc_loss()
        
        total_loss = task_loss + ewc_loss
        total_loss.backward()
        
        self.optimizer.step()
        
        return total_loss.item()

    def update_fisher_information(self, new_task_data):
        self.calculate_fisher_information(new_task_data)

    def adapt_to_new_task(self, new_task_data, num_epochs=10):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in new_task_data:
                loss = self.train_step(batch)
                epoch_loss += loss
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(new_task_data):.4f}")
        
        self.update_fisher_information(new_task_data)
