import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        super().__init__()

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w0 = nn.Parameter(torch.tensor(0.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x):
        return self.w0 + self.w1 * x

    def fit(self, X_train, y_train, X_test, y_test):
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X_train)
            # Calculate loss
            loss = self.loss_fn(y_pred, y_train)

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # update weights
            self.optimizer.step()

            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        with torch.no_grad():
            y_test_pred = self.forward(X_test)

            ss_res = torch.sum((y_test - y_test_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)

            r2 = 1 - (ss_res / ss_tot)

            print(f"R^2 score: {r2.item():.4f}")

    def predict(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=self.w0.dtype)
            return self.forward(x)

    def analysis_plot(self, X_train, y_train):
        X = X_train.detach().numpy()
        y = y_train.detach().numpy()

        with torch.no_grad():
            y_pred = self.forward(X_train).detach().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Original data and fitted regression line
        axs[0, 0].scatter(X, y, label="Original Data")
        axs[0, 0].plot(
            X.flatten()[X.flatten().argsort()],
            y_pred.flatten()[X.flatten().argsort()],
            color="red",
            label="Regression Line"
        )
        axs[0, 0].set_title("Linear Regression")
        axs[0, 0].set_xlabel("BCR")
        axs[0, 0].set_ylabel("Annual Production")
        axs[0, 0].legend()

        # w0 history
        axs[0, 1].plot(self.w0_history)
        axs[0, 1].set_title("w0 Training History")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("w0")

        # w1 history
        axs[1, 0].plot(self.w1_history)
        axs[1, 0].set_title("w1 Training History")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("w1")

        # Loss history
        axs[1, 1].plot(self.loss_history)
        axs[1, 1].set_title("Training Loss")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("MSE")

        plt.tight_layout()
        plt.show()