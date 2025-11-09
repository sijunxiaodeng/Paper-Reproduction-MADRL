# Defender 

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
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

```
输入的state：是参考过去T(超参数)轮的

T * (M*N + L*N)：M/L：防御者/攻击者数目；N：主机数

# Novel Strategy-Based Sampling Approach
从经验池中取样时，先计算平均值，再计算欧拉距离按照降序选取m个

# Attacker

## policy function
依据重要性uk和先前防御者策略
首先计算前t轮防御者分配给主机i的平均值，并且按照uk的优先级分配攻击资源，且大于前t轮防御者部署的平均资源
