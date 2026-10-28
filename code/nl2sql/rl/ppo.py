"""PPO (primary) / A2C (ablation) on the cached-tree environment, CPU only. Actor and critic MLPs 27->64->64 (tanh),
Adam lr 1e-3 annealed to 1e-4, entropy 0.01, clip 0.2, 4 epochs x minibatch 256, 2048 env steps per update, 200 updates,
GAE lambda 0.95, gamma 1, invalid-action masking."""
import numpy as np
import torch
import torch.nn as nn

from .env import VecEnv, evaluate_policy
from .features import STATE_DIM

torch.set_num_threads(1)


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(STATE_DIM, 5)
        self.critic = mlp(STATE_DIM, 1)

    def dist(self, s, m):
        lg = self.actor(s).masked_fill(~m, -1e9)
        return torch.distributions.Categorical(logits=lg)

    def v(self, s):
        return self.critic(s).squeeze(-1)


def export_npz(model, path):
    a = model.actor
    np.savez(path, w1=a[0].weight.detach().numpy().T, b1=a[0].bias.detach().numpy(), w2=a[2].weight.detach().numpy().T,
             b2=a[2].bias.detach().numpy(), w3=a[4].weight.detach().numpy().T, b3=a[4].bias.detach().numpy())


def greedy_policy(model):
    def pol(s, mask):
        with torch.no_grad():
            lg = model.actor(torch.as_tensor(s).float().unsqueeze(0))[0].numpy()
        return int(np.argmax(np.where(mask, lg, -1e9)))
    return pol


def train(ts, c_h, lam, seed=0, algo="ppo", updates=200, steps=2048, n_envs=32, a4=True, q_subset=None, log=None,
          gae_lambda=0.95, clip=0.2, ent=0.01, epochs=4, mb=256, lr0=1e-3, lr1=1e-4):
    torch.manual_seed(seed)
    rs = np.random.RandomState(seed)
    env = VecEnv(ts, n_envs, c_h, lam, a4=a4, seed=seed, q_subset=q_subset)
    model = AC()
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    T = steps // n_envs
    rets = []
    for u in range(updates):
        for g in opt.param_groups:
            g["lr"] = lr0 + (lr1 - lr0) * u / max(1, updates - 1)
        S = np.zeros((T, n_envs, STATE_DIM), np.float32)
        M = np.zeros((T, n_envs, 5), bool)
        A = np.zeros((T, n_envs), np.int64)
        LP = np.zeros((T, n_envs), np.float32)
        V = np.zeros((T, n_envs), np.float32)
        R = np.zeros((T, n_envs), np.float32)
        D = np.zeros((T, n_envs), np.float32)
        ep_ret = np.zeros(n_envs)
        ep_done_rets = []
        for t in range(T):
            s, m = env.obs(), env.masks()
            with torch.no_grad():
                st, mt = torch.as_tensor(s), torch.as_tensor(m)
                d = model.dist(st, mt)
                a = d.sample()
                LP[t], V[t] = d.log_prob(a).numpy(), model.v(st).numpy()
            S[t], M[t], A[t] = s, m, a.numpy()
            r, done = env.step(A[t])
            R[t], D[t] = r, done
            ep_ret += r
            for i in np.where(done)[0]:
                ep_done_rets.append(ep_ret[i])
                ep_ret[i] = 0
        with torch.no_grad():
            last_v = model.v(torch.as_tensor(env.obs())).numpy()
        adv = np.zeros_like(R)
        gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nd = 1.0 - D[t]
            delta = R[t] + nv * nd - V[t]
            gae = delta + gae_lambda * nd * gae
            adv[t] = gae
        ret = adv + V
        fl = lambda x: torch.as_tensor(x.reshape(T * n_envs, *x.shape[2:]))
        bS, bM, bA, bLP, bAdv, bRet = fl(S), fl(M), fl(A), fl(LP), fl(adv), fl(ret)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)
        N = T * n_envs
        for _ in range(epochs if algo == "ppo" else 1):
            idx = rs.permutation(N) if algo == "ppo" else np.arange(N)
            for s0 in range(0, N, mb if algo == "ppo" else N):
                j = torch.as_tensor(idx[s0:s0 + (mb if algo == "ppo" else N)])
                d = model.dist(bS[j], bM[j])
                lp = d.log_prob(bA[j])
                if algo == "ppo":
                    ratio = torch.exp(lp - bLP[j])
                    pl = -torch.min(ratio * bAdv[j], torch.clamp(ratio, 1 - clip, 1 + clip) * bAdv[j]).mean()
                else:
                    pl = -(lp * bAdv[j]).mean()
                vl = ((model.v(bS[j]) - bRet[j]) ** 2).mean()
                loss = pl + 0.5 * vl - ent * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        rets.append(float(np.mean(ep_done_rets)) if ep_done_rets else float("nan"))
        if log and (u % 20 == 0 or u == updates - 1):
            log("update %d mean_ep_return %.3f" % (u, rets[-1]))
    tail = [x for x in rets[int(0.8 * updates):] if x == x]
    return model, {"train_return_last20": float(np.mean(tail)) if tail else None, "returns": rets}
