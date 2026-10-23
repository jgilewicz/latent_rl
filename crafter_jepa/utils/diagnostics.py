import torch


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_returns = returns.var()
    if var_returns < 1e-8:
        return 0.0
    return (1 - (returns - values).var() / var_returns).item()


def effective_rank(features: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(features - features.mean(dim=0))
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return torch.exp(entropy).item()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cross = torch.linalg.matrix_norm(x.T @ y) ** 2
    norm = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return (cross / norm.clamp_min(1e-12)).item()


class OnlineProbe:
    def __init__(
        self, feature_dim: int, n_labels: int, device: torch.device, lr: float = 1e-3
    ) -> None:
        self.head = torch.nn.Linear(feature_dim, n_labels).to(device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

    def fit_and_score(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 256,
        holdout: float = 0.2,
    ) -> dict[str, float]:
        features = features.detach()
        perm = torch.randperm(len(features), device=features.device)
        n_test = int(len(features) * holdout)
        test, train = perm[:n_test], perm[n_test:]

        for start in range(0, len(train), batch_size):
            idx = train[start : start + batch_size]
            loss = torch.nn.functional.mse_loss(self.head(features[idx]), labels[idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            pred = self.head(features[test])
            target = labels[test]
            sse = ((pred - target) ** 2).sum()
            sst = ((target - target.mean(dim=0)) ** 2).sum()
            return {
                "probe/mse": ((pred - target) ** 2).mean().item(),
                "probe/r2": (1 - sse / sst).item() if sst > 1e-6 else 0.0,
            }
