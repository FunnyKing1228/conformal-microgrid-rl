import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import math


class EvidentialHead(nn.Module):
    """
    Evidential Uncertainty Head
    
    實現Normal-Inverse-Gamma (NIG) 分佈的頭部，用於量化不確定性
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.1,
        gamma_regularizer: float = 1e-2,
        epsilon: float = 1e-6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gamma_regularizer = gamma_regularizer
        self.epsilon = epsilon
        
        # 激活函數
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 共享特徵提取層
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            self.activation
        )
        
        # NIG參數輸出層
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)  # 均值
        self.log_lambda_head = nn.Linear(hidden_dim // 2, output_dim)  # log(λ)
        self.log_alpha_head = nn.Linear(hidden_dim // 2, output_dim)  # log(α-1)
        self.log_beta_head = nn.Linear(hidden_dim // 2, output_dim)  # log(β)
        
        # 初始化權重
        self._init_weights()
        
        print(f"✓ EvidentialHead initialized:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Output dimension: {output_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Gamma regularizer: {gamma_regularizer}")
        print(f"  - Epsilon: {epsilon}")
    
    def _init_weights(self):
        """初始化網絡權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch_size, input_dim]
            
        Returns:
            (mu, lambda, alpha, beta): NIG分佈參數
        """
        # 特徵提取
        features = self.feature_extractor(x)
        
        # 輸出NIG參數
        mu = self.mu_head(features)  # 均值
        log_lambda = self.log_lambda_head(features)  # log(λ)
        log_alpha_minus_1 = self.log_alpha_head(features)  # log(α-1)
        log_beta = self.log_beta_head(features)  # log(β)
        
        # 應用softplus確保參數為正
        lambda_val = F.softplus(log_lambda) + self.epsilon
        alpha = F.softplus(log_alpha_minus_1) + 1.0 + self.epsilon  # α > 1
        beta = F.softplus(log_beta) + self.epsilon
        
        return mu, lambda_val, alpha, beta
    
    def loss(self, y: torch.Tensor, mu: torch.Tensor, lambda_val: torch.Tensor, 
             alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算Evidential損失
        
        Args:
            y: 真實標籤
            mu, lambda_val, alpha, beta: NIG參數
            
        Returns:
            (total_loss, loss_components): 總損失和損失組件
        """
        # 計算NLL損失
        nll_loss = self._nll_loss(y, mu, lambda_val, alpha, beta)
        
        # 計算正則化損失
        reg_loss = self._regularization_loss(y, mu, lambda_val, alpha)
        
        # 總損失
        total_loss = nll_loss + self.gamma_regularizer * reg_loss
        
        # 損失組件
        loss_components = {
            'nll_loss': nll_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _nll_loss(self, y: torch.Tensor, mu: torch.Tensor, lambda_val: torch.Tensor, 
                   alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        計算Negative Log-Likelihood損失
        """
        # 計算NIG的NLL
        # 參考: https://arxiv.org/abs/1807.00263
        
        # 計算中間變量
        lambda_term = lambda_val * (y - mu) ** 2
        beta_term = 2 * beta + lambda_term
        
        # NLL計算
        log_term = torch.log(lambda_val / (2 * math.pi))
        alpha_term = alpha * torch.log(beta_term / (2 * beta))
        beta_term_log = torch.log(beta_term)
        
        # 組合NLL
        nll = 0.5 * (log_term - alpha_term + beta_term_log)
        
        return nll.mean()
    
    def _regularization_loss(self, y: torch.Tensor, mu: torch.Tensor, 
                            lambda_val: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        計算正則化損失，防止過度自信
        """
        # 基於預測誤差的正則化
        prediction_error = torch.abs(y - mu)
        evidence_penalty = prediction_error * (2 * lambda_val + alpha)
        
        return evidence_penalty.mean()
    
    def get_uncertainty(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                        alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        計算不確定性指標
        
        Args:
            mu, lambda_val, alpha, beta: NIG參數
            
        Returns:
            (aleatoric, epistemic): 隨機不確定性和認知不確定性
        """
        # 隨機不確定性 (Aleatoric): E[σ²] = β / (α - 1)
        aleatoric = beta / (alpha - 1)
        
        # 認知不確定性 (Epistemic): Var(μ) = β / (λ(α - 1))
        epistemic = beta / (lambda_val * (alpha - 1))
        
        return aleatoric, epistemic
    
    def get_total_uncertainty(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                              alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        計算總不確定性
        
        Args:
            mu, lambda_val, alpha, beta: NIG參數
            
        Returns:
            total_uncertainty: 總不確定性
        """
        aleatoric, epistemic = self.get_uncertainty(mu, lambda_val, alpha, beta)
        total_uncertainty = torch.sqrt(aleatoric + epistemic)
        return total_uncertainty
    
    def sample_predictions(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                           alpha: torch.Tensor, beta: torch.Tensor, 
                           num_samples: int = 100) -> torch.Tensor:
        """
        從NIG分佈採樣預測值
        
        Args:
            mu, lambda_val, alpha, beta: NIG參數
            num_samples: 採樣數量
            
        Returns:
            samples: 採樣的預測值 [num_samples, batch_size, output_dim]
        """
        batch_size = mu.shape[0]
        output_dim = mu.shape[1]
        
        # 從Inverse-Gamma分佈採樣σ²
        sigma_squared = torch.distributions.InverseGamma(alpha, beta).sample([num_samples])
        sigma_squared = sigma_squared.transpose(0, 1)  # [batch_size, num_samples, output_dim]
        
        # 從Normal分佈採樣預測值
        sigma = torch.sqrt(sigma_squared)
        samples = torch.distributions.Normal(mu.unsqueeze(1), sigma / torch.sqrt(lambda_val.unsqueeze(1))).sample()
        
        return samples.transpose(0, 1)  # [num_samples, batch_size, output_dim]


class EvidentialCritic(nn.Module):
    """
    帶有Evidential頭的Critic網絡
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        evidential: bool = True,
        gamma_regularizer: float = 1e-2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.evidential = evidential
        self.gamma_regularizer = gamma_regularizer
        
        # 共享特徵提取層
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if evidential:
            # Evidential頭
            self.evidential_head = EvidentialHead(
                input_dim=hidden_dim,
                output_dim=1,
                hidden_dim=hidden_dim // 2,
                gamma_regularizer=gamma_regularizer
            )
        else:
            # 標準Q值頭
            self.q_head = nn.Linear(hidden_dim, 1)
        
        print(f"✓ EvidentialCritic initialized:")
        print(f"  - State dimension: {state_dim}")
        print(f"  - Action dimension: {action_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Evidential: {evidential}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            state: 狀態張量
            action: 動作張量
            
        Returns:
            Q值或NIG參數
        """
        # 連接狀態和動作
        x = torch.cat([state, action], dim=-1)
        
        # 特徵提取
        features = self.feature_net(x)
        
        if self.evidential:
            # 返回NIG參數
            mu, lambda_val, alpha, beta = self.evidential_head(features)
            return mu, lambda_val, alpha, beta
        else:
            # 返回Q值
            q_value = self.q_head(features)
            return q_value
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        獲取Q值（用於標準SAC更新）
        """
        if self.evidential:
            mu, _, _, _ = self.forward(state, action)
            return mu
        else:
            return self.forward(state, action)
    
    def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        獲取不確定性（僅在evidential模式下有效）
        """
        if not self.evidential:
            return torch.zeros(state.shape[0], 1, device=state.device)
        
        mu, lambda_val, alpha, beta = self.forward(state, action)
        return self.evidential_head.get_total_uncertainty(mu, lambda_val, alpha, beta)


def create_evidential_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    evidential: bool = True,
    gamma_regularizer: float = 1e-2
) -> EvidentialCritic:
    """
    創建EvidentialCritic的工廠函數
    
    Args:
        state_dim: 狀態維度
        action_dim: 動作維度
        hidden_dim: 隱藏層維度
        evidential: 是否使用evidential頭
        gamma_regularizer: 正則化係數
        
    Returns:
        EvidentialCritic: 配置好的critic網絡
    """
    return EvidentialCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        evidential=evidential,
        gamma_regularizer=gamma_regularizer
    ) 