import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict, Any
import gymnasium as gym
import math


class Actor(nn.Module):
    """Actor network for SAC - outputs mean and log_std for Gaussian policy"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))  # Output in [-1, 1]
        log_std = torch.clamp(self.fc_logstd(x), -20, 2)  # Clamp log_std
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Tanh squashing
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC - outputs Q-value for state-action pair"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OccHead(nn.Module):
    """Opportunity-Cost head: predicts proxy cost ~ Δa/Pmax given (s,a)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class EvidentialHead(nn.Module):
    """Outputs Normal-Inverse-Gamma parameters (mu, nu, alpha, beta)"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, 1)
        self.nu = nn.Linear(in_dim, 1)
        self.alpha = nn.Linear(in_dim, 1)
        self.beta = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        nu = F.softplus(self.nu(x)) + 1e-6
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1e-6
        return mu, nu, alpha, beta


class EvidentialCritic(nn.Module):
    """Critic that outputs evidential parameters instead of scalar Q"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = EvidentialHead(hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


def nig_nll(y: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for Normal-Inverse-Gamma evidential regression"""
    eps = 1e-6
    term1 = 0.5 * torch.log(math.pi / (nu + eps))
    term2 = -alpha * torch.log(2 * beta * (1 + nu) + eps)
    term3 = (alpha + 0.5) * torch.log(nu * (y - mu).pow(2) + 2 * beta * (1 + nu) + eps)
    term4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return term1 + term2 + term3 + term4


def evidential_reg(y: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    return lam * torch.abs(y - mu) * (2 * nu + alpha)


class SACAgent:
    """Soft Actor-Critic agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 1e-4,          # ★ alpha 獨立 lr（比 actor 低）
        gamma: float = 0.99,
        tau: float = 5e-3,
        alpha: float = 0.2,
        target_entropy: float = -1.0,
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        batch_size: int = 256,
        evidential_enabled: bool = False,
        lambda_evi: float = 1e-3,
        beta_risk: float = 0.5,
        beta_occ: float = 0.3
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy
        self.batch_size = batch_size
        self.evidential_enabled = evidential_enabled
        self.lambda_evi = lambda_evi
        self.beta_risk = beta_risk
        self.beta_occ = beta_occ
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        if self.evidential_enabled:
            self.critic1 = EvidentialCritic(state_dim, action_dim, hidden_dim).to(device)
            self.critic2 = EvidentialCritic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic1 = EvidentialCritic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic2 = EvidentialCritic(state_dim, action_dim, hidden_dim).to(device)
        else:
            self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
            self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
            self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        # OCC head and optimizer
        self.occ_head = OccHead(state_dim, action_dim, hidden_dim).to(device)
        self.occ_optimizer = optim.Adam(self.occ_head.parameters(), lr=lr_critic)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Log alpha for automatic entropy adjustment
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            mean, _ = self.actor(state)
            action = mean
        else:
            action, _ = self.actor.sample(state)
            
        return action.detach().cpu().numpy()[0]
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool,
                        occ_proxy: float = 0.0):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done, float(occ_proxy))
    
    def update(self) -> Dict[str, float]:
        """Update networks using one batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Sample batch
        states, actions, rewards, next_states, dones, occ_proxies = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        if not self.evidential_enabled:
            # Standard SAC updates (MSE critics)
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q

            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)

            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # OCC head update（監督 Δa/Pmax 代理成本）
            occ_target = torch.FloatTensor(occ_proxies).unsqueeze(1).to(self.device)
            occ_pred = self.occ_head(states, actions)
            occ_loss = F.mse_loss(occ_pred, occ_target)
            self.occ_optimizer.zero_grad()
            occ_loss.backward()
            self.occ_optimizer.step()

            # Actor update（加入 OCC 規則，鼓勵小機會成本）
            actions_pi, log_probs_pi = self.actor.sample(states)
            q1_pi = self.critic1(states, actions_pi)
            q2_pi = self.critic2(states, actions_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            occ_pred_pi = self.occ_head(states, actions_pi).detach()
            actor_loss = (self.log_alpha.exp() * log_probs_pi - q_pi + self.beta_occ * occ_pred_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Alpha update（含 clamp 防止爆炸）
            alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # ★ Clamp log_alpha: exp(-5)≈0.007, exp(2)≈7.4
            with torch.no_grad():
                self.log_alpha.clamp_(min=-5.0, max=2.0)
            self.alpha = self.log_alpha.exp()

            # Soft update targets
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

            out = {
                'critic1_loss': float(critic1_loss.item()),
                'critic2_loss': float(critic2_loss.item()),
                'actor_loss': float(actor_loss.item()),
                'alpha_loss': float(alpha_loss.item()),
                'alpha': float(self.alpha.item()),
                'occ_loss': float(F.mse_loss(occ_pred.detach(), occ_target).item())
            }
            return out
        else:
            # Evidential critics and risk-aware actor
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                mu1_t, nu1_t, a1_t, b1_t = self.target_critic1(next_states, next_actions)
                mu2_t, nu2_t, a2_t, b2_t = self.target_critic2(next_states, next_actions)
                q_next = torch.min(mu1_t, mu2_t) - self.log_alpha.exp() * next_log_probs
                y = rewards + (1 - dones) * self.gamma * q_next

            mu1, nu1, a1, b1 = self.critic1(states, actions)
            mu2, nu2, a2, b2 = self.critic2(states, actions)

            c1_nll = nig_nll(y, mu1, nu1, a1, b1)
            c2_nll = nig_nll(y, mu2, nu2, a2, b2)
            c1_reg = evidential_reg(y, mu1, nu1, a1, lam=self.lambda_evi)
            c2_reg = evidential_reg(y, mu2, nu2, a2, lam=self.lambda_evi)
            critic1_loss = (c1_nll + c1_reg).mean()
            critic2_loss = (c2_nll + c2_reg).mean()

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10.0)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
            self.critic2_optimizer.step()

            # Actor update with risk-aware Q
            actions_pi, log_probs_pi = self.actor.sample(states)
            mu1_pi, nu1_pi, A1_pi, B1_pi = self.critic1(states, actions_pi)
            mu2_pi, nu2_pi, A2_pi, B2_pi = self.critic2(states, actions_pi)
            sigma1 = torch.sqrt(B1_pi / (nu1_pi * (A1_pi - 1.0).clamp_min(1e-6) + 1e-8))
            sigma2 = torch.sqrt(B2_pi / (nu2_pi * (A2_pi - 1.0).clamp_min(1e-6) + 1e-8))
            mu_pi = torch.min(mu1_pi, mu2_pi)
            sigma_pi = torch.max(sigma1, sigma2)
            q_risk = mu_pi - self.beta_risk * sigma_pi
            actor_loss = (self.log_alpha.exp() * log_probs_pi - q_risk).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Alpha update (no sigma-guided tweak here, 含 clamp)
            alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=-5.0, max=2.0)
            self.alpha = self.log_alpha.exp()

            # Soft update targets
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

            return {
                'critic1_loss': float(critic1_loss.item()),
                'critic2_loss': float(critic2_loss.item()),
                'actor_loss': float(actor_loss.item()),
                'alpha_loss': float(alpha_loss.item()),
                'alpha': float(self.alpha.item()),
            }
     
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'occ_head': self.occ_head.state_dict(),
            'occ_optimizer': self.occ_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'evidential_enabled': self.evidential_enabled,
            'lambda_evi': self.lambda_evi,
            'beta_risk': self.beta_risk,
            'beta_occ': self.beta_occ,
        }, path)
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if 'occ_head' in checkpoint:
            self.occ_head.load_state_dict(checkpoint['occ_head'])
        if 'occ_optimizer' in checkpoint:
            self.occ_optimizer.load_state_dict(checkpoint['occ_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']
        # 可選：恢復 evidential 超參
        if 'evidential_enabled' in checkpoint:
            self.evidential_enabled = checkpoint['evidential_enabled']
        if 'lambda_evi' in checkpoint:
            self.lambda_evi = float(checkpoint['lambda_evi'])
        if 'beta_risk' in checkpoint:
            self.beta_risk = float(checkpoint['beta_risk'])
        if 'beta_occ' in checkpoint:
            self.beta_occ = float(checkpoint['beta_occ'])


class ReplayBuffer:
    """Simple experience replay buffer"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, occ_proxy: float = 0.0):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done, float(occ_proxy)))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, occs = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), np.array(occs))
    
    def __len__(self) -> int:
        return len(self.buffer) 