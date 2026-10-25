import numpy as np
import torch


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_returns = returns.var()
    if var_returns < 1e-8:
        return 0.0
    return (1 - (returns - values).var() / var_returns).item()


def effective_rank(features: torch.Tensor) -> float:
    centered = (features - features.mean(dim=0)).double()
    # eigvalsh of the d×d Gram is ~10× faster than svdvals of the n×d matrix;
    # float64 keeps it equal to svdvals
    singular = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0).sqrt()
    probs = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return torch.exp(entropy).item()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cross = torch.linalg.matrix_norm(x.T @ y) ** 2
    norm = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return (cross / norm.clamp_min(1e-12)).item()


def linear_cka_matrix(latents: list[torch.Tensor]) -> torch.Tensor:
    centered = [z - z.mean(dim=0) for z in latents]
    grams = [[x.T @ y for y in centered] for x in centered]
    norms = torch.stack(
        [torch.linalg.matrix_norm(grams[i][i]) for i in range(len(grams))]
    )
    cross = torch.stack(
        [torch.stack([torch.linalg.matrix_norm(g) ** 2 for g in row]) for row in grams]
    )
    return cross / torch.outer(norms, norms).clamp_min(1e-12)


def nn_distance_ratios(
    latents: torch.Tensor,
    episode: torch.Tensor,
    t: torch.Tensor,
    theiler_window: int,
    chunk: int = 4096,
) -> torch.Tensor:
    # float32 rounds small first-neighbour distances in ‖a‖²+‖b‖²−2ab to 0
    z = latents.double()
    sq_norms = (z * z).sum(dim=1)
    nearest = []
    for start in range(0, len(z), chunk):
        rows = slice(start, start + chunk)
        dist2 = sq_norms[rows, None] + sq_norms[None] - 2 * z[rows] @ z.T
        same_episode = episode[rows, None] == episode[None]
        close_in_time = (t[rows, None] - t[None]).abs() < theiler_window
        dist2[same_episode & close_in_time] = float("inf")
        nearest.append(dist2.topk(2, largest=False).values.clamp_min(0).sqrt())
    r = torch.cat(nearest)
    return r[:, 1] / r[:, 0]


def two_nn_dimension(ratios: np.ndarray, discard: float = 0.1) -> float:
    mu = np.sort(ratios[~np.isnan(ratios)])
    cdf = np.arange(1, len(mu) + 1) / len(mu)
    n_fit = int(len(mu) * (1 - discard))
    x, y = np.log(mu[:n_fit]), -np.log(1 - cdf[:n_fit])
    return float((x * y).sum() / (x * x).sum())


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
