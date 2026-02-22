import numpy as np
import torch


class PretrainedLinUCB:
    def __init__(self, feature_dim, observed_dim, inference_model, alpha=1.0):
        self.d = feature_dim
        self.observed_d = observed_dim
        self.inf_model = inference_model
        self.alpha = alpha
        self.A = np.identity(self.d)
        self.b = np.zeros(self.d)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.inf_model.to(self.device)
        self.inf_model.eval()

    def _predict_full_context(self, s_observed):
        if s_observed.ndim == 1:
            s_observed = s_observed.reshape(1, -1)

        with torch.no_grad():
            s_tensor = torch.from_numpy(s_observed).float().to(self.device)
            s_prime_pred_tensor = self.inf_model(s_tensor)
            s_prime_predicted = s_prime_pred_tensor.cpu().numpy()

        return np.hstack([s_observed, s_prime_predicted])

    def select_arm(self, arm_features_observed):
        reconstructed_features = self._predict_full_context(arm_features_observed)
        a_inv = np.linalg.inv(self.A)
        theta = a_inv.dot(self.b)
        ucb_scores = [
            theta.dot(x_a) + self.alpha * np.sqrt(x_a.dot(a_inv).dot(x_a))
            for x_a in reconstructed_features
        ]
        return int(np.argmax(ucb_scores))

    def update(self, chosen_arm_feature_observed, reward):
        x_chosen_hat = self._predict_full_context(chosen_arm_feature_observed).flatten()
        self.A += np.outer(x_chosen_hat, x_chosen_hat)
        self.b += reward * x_chosen_hat


class BaselineLinUCB:
    def __init__(self, feature_dim, alpha=1.0):
        self.d = feature_dim
        self.alpha = alpha
        self.A = np.identity(self.d)
        self.b = np.zeros(self.d)

    def select_arm(self, arm_features):
        a_inv = np.linalg.inv(self.A)
        theta = a_inv.dot(self.b)
        ucb_scores = [
            theta.dot(x_a) + self.alpha * np.sqrt(x_a.dot(a_inv).dot(x_a))
            for x_a in arm_features
        ]
        return int(np.argmax(ucb_scores))

    def update(self, chosen_arm_feature, reward):
        self.A += np.outer(chosen_arm_feature, chosen_arm_feature)
        self.b += reward * chosen_arm_feature
