import numpy as np
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

class GaussianProcessRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcessRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Training Data
def generate_training_data(n=20):
    train_x = torch.linspace(0, 1, n)
    train_y = torch.sin(train_x * (2 * np.pi)) + 0.2 * torch.randn(train_x.size())
    return train_x, train_y

# Train GP Model
def train_gp_model(train_x, train_y, training_iterations=50):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GaussianProcessRegressionModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    return model, likelihood

# Sampling from GP Prior
def sample_from_gp_prior(n_samples=5, x_range=(-3, 3), n_points=100):
    x = torch.linspace(*x_range, n_points).unsqueeze(1)
    kernel = ScaleKernel(RBFKernel())
    cov_matrix = kernel(x).evaluate() + 1e-6 * torch.eye(n_points)
    samples = torch.distributions.MultivariateNormal(torch.zeros(n_points), cov_matrix).sample((n_samples,))
    return x, samples

if __name__ == "__main__":
    train_x, train_y = generate_training_data()
    model, likelihood = train_gp_model(train_x, train_y)
    x, samples = sample_from_gp_prior()
