# Defender Class

## DQN
```commandline
  def init_dqn(self, state_dim, action_dim, lr=0.001):
        """初始化DQN网络"""
        self.dqn = nn.Sequential(
            nn.Linear(state_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, action_dim)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

```
state_dim状态：state_dim的核心逻辑是：将 T 轮历史中所有攻防策略的信息扁平化后得到的总长度，公式为：state_dim = T * (防御者总数×主机数 + 攻击者总数×主机数)
