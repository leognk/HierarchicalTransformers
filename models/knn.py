import torch
import torch.distributed
from utils import ArgumentParser, get_args


CONFIGS_ROOT = "configs/models/knn"


class KNN:

    def __init__(self, n_classes, k, temperature, batch_size):
        self.n_classes = n_classes
        self.k = k
        self.temperature = temperature
        self.batch_size = batch_size
        self.reset()
    
    def reset(self):
        self.train_features = []
        self.train_targets = []
        self.eval_features = []
        self.eval_targets = []
        self.train_features_ready = False
        self.eval_features_ready = False

    @staticmethod
    def global_pool(x):
        """shape: [b * d] -> [b d]"""
        if x.dim() > 2:
            x = torch.movedim(x, -1, 1)
            x = torch.mean(torch.flatten(x, start_dim=2), dim=-1)
        return x
    
    def _add_features(self, x, y, features, targets):
        """
        x: [b * d]
        y: [b]
        """
        x = self.global_pool(x) # [b * d] -> [b d]
        features.append(x)
        targets.append(y)
    
    def add_train_features(self, x, y):
        assert not self.train_features_ready
        self._add_features(x, y, self.train_features, self.train_targets)
    
    def add_eval_features(self, x, y):
        assert not self.eval_features_ready
        self._add_features(x, y, self.eval_features, self.eval_targets)
    
    def _gather_features(self, ddp, x, y):
        x = torch.cat(x)
        y = torch.cat(y)

        if not ddp.use_ddp: return x, y

        bs = torch.tensor([x.shape[0]], dtype=torch.int, device=x.device)
        all_bs = [torch.empty(1, dtype=torch.int, device=x.device) for _ in range(ddp.world_size)]
        torch.distributed.all_gather(all_bs, bs)

        all_x = [torch.empty(b.item(), x.shape[1], dtype=x.dtype, device=x.device) for b in all_bs]
        torch.distributed.all_gather(all_x, x)

        all_y = [torch.empty(b.item(), dtype=y.dtype, device=y.device) for b in all_bs]
        torch.distributed.all_gather(all_y, y)

        return torch.cat(all_x), torch.cat(all_y)
    
    def gather_train_features(self, ddp):
        assert not self.train_features_ready
        self.train_features, self.train_targets = self._gather_features(ddp, self.train_features, self.train_targets)
        self.train_features = torch.nn.functional.normalize(self.train_features, dim=1)
        self.train_features_ready = True
    
    def gather_eval_features(self, ddp):
        assert not self.eval_features_ready
        self.eval_features, self.eval_targets = self._gather_features(ddp, self.eval_features, self.eval_targets)
        self.eval_features = torch.nn.functional.normalize(self.eval_features, dim=1)
        self.eval_features_ready = True
    
    def classify(self):
        assert self.train_features_ready and self.eval_features_ready

        acc1, acc5 = 0, 0
        n_eval = self.eval_targets.shape[0]

        for i in range(0, n_eval, self.batch_size):
            # Get the features for test samples.
            i1 = i + self.batch_size
            features = self.eval_features[i:i1] # [m d]
            targets = self.eval_targets[i:i1] # [m]
            m = targets.shape[0]

            # Calculate the dot products and the k nearest neighbors.
            sim = torch.matmul(features, self.train_features.T) # [m n]
            topk_sim, topk_ids = sim.topk(self.k) # [m k], [m k]
            topk_labels = torch.gather(self.train_targets.expand(m, -1), dim=1, index=topk_ids) # [m k]

            # Calculate the scores and the predictions in sorted order.
            topk_scores = torch.exp(topk_sim / self.temperature) # [m k]
            scores = torch.zeros(m, self.n_classes, dtype=topk_scores.dtype, device=sim.device) # [m c]
            scores.scatter_reduce_(dim=1, index=topk_labels, src=topk_scores, reduce='sum') # [m c]
            preds = scores.argsort(descending=True) # [m c]

            # Compute number of top k correct predictions.
            correct = preds == targets.reshape(-1, 1) # [m c]
            acc1 += correct[:, :1].sum().item()
            acc5 += correct[:, :min(5, self.k)].sum().item() # top 5 does not make sense if k < 5
        
        acc1 = 100 * acc1 / n_eval
        acc5 = 100 * acc5 / n_eval
        return acc1, acc5


def get_args_parser():
    parser = ArgumentParser("knn", add_help=False)
    parser.add_argument("--k", type=int)
    parser.add_argument("--temperature", type=float)
    return parser


def create_knn(config, n_classes, batch_size):
    args = get_args(CONFIGS_ROOT, (get_args_parser(), config))
    knn = KNN(n_classes, args.k, args.temperature, batch_size)
    knn.args = args
    return knn