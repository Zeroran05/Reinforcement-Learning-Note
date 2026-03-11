# 本代码对最基本的PPO算法进行了验证，使用CartPole-v1环境进行训练和测试。
# 代码使用的是PPO_Clip版本，原论文还有KL版本；
# 本代码的PPO算法是on-policy的，想实现off-policy，可以通过重要性采样/经验回放实现。

import torch  # Import PyTorch, a popular machine learning library
import torch.nn as nn  # Import the neural network module
import torch.optim as optim  # Import optimization algorithms
from torch.distributions import Categorical  # Import Categorical for probabilistic action sampling
import numpy as np  # Import NumPy for numerical computations
import gymnasium as gym # Import OpenAI Gym for environment simulation
import matplotlib.pyplot as plt

# Define Actor-Critic Network
class ActorCritic(nn.Module):  # Define the Actor-Critic model 定义Actor-Critic子类，继承自nn.Module类
    def __init__(self, state_dim, action_dim):  # Initialize with state and action dimensions 构造方法，初始化状态维度和动作维度
        super(ActorCritic, self).__init__()  # Call parent class constructor 调用父类构造方法
        self.shared_layer = nn.Sequential(  # Shared network layers for feature extraction 定义共享层，用于特征提取
            nn.Linear(state_dim, 128),  # Fully connected layer with 128 neurons 定义全连接层，输入维度为状态维度，输出维度为128
            nn.ReLU()  # ReLU activation function ReLU激活函数
        )
        self.actor = nn.Sequential(  # Define the actor (policy) network
            nn.Linear(128, action_dim),  # Fully connected layer to output action probabilities
            nn.Softmax(dim=-1)  # Softmax to ensure output is a probability distribution
        )
        self.critic = nn.Linear(128, 1)  # Define the critic (value) network to output state value

    def forward(self, state):  # Forward pass for the model
        shared = self.shared_layer(state)  # Pass state through shared layers
        action_probs = self.actor(shared)  # Get action probabilities from actor network
        state_value = self.critic(shared)  # Get state value from critic network
        return action_probs, state_value  # Return action probabilities and state value

# Memory to store experiences
class Memory:  # Class to store agent's experience
    def __init__(self):  # Initialize memory
        self.states = []  # List to store states
        self.actions = []  # List to store actions
        self.logprobs = []  # List to store log probabilities of actions
        self.rewards = []  # List to store rewards
        self.is_terminals = []  # List to store terminal state flags

    def clear(self):  # Clear memory after an update
        self.states = []  # Clear stored states
        self.actions = []  # Clear stored actions
        self.logprobs = []  # Clear stored log probabilities
        self.rewards = []  # Clear stored rewards
        self.is_terminals = []  # Clear terminal state flags

# PPO Agent
class PPO:  # Define the PPO agent
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim).to(device)  # 创建Actor-Critic网络，并将其移动到指定设备（CPU或GPU）
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # 使用Adam优化器优化策略网络的参数，学习率为lr,这里就指定了更新的是哪个策略了
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)  # Copy of the policy for stability
        self.policy_old.load_state_dict(self.policy.state_dict())  # 将当前策略的参数复制到旧策略中
        self.MseLoss = nn.MSELoss()  # Mean Squared Error loss for critic updates

        self.gamma = gamma  # Discount factor for rewards
        self.eps_clip = eps_clip  # Clipping parameter for PPO
        self.K_epochs = K_epochs  # Number of epochs for optimization
        
        # 记录loss的历史，以便后续分析和可视化
        self.loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        
        

    def select_action(self, state, memory):  #duck typeing: 只要传入的state和memory满足底下的要求，就可以使用这个方法，不需要关心它们的具体类型,
                                             #进入程序才会检查,故可以在没有声明/实例化Memory类的情况下使用类下的属性.
        state = torch.FloatTensor(state).to(device)  # 将输入状态转换为PyTorch张量，并移动到指定设备
        action_probs, _ = self.policy_old(state)  # 实际中在pytorch下会调用_call_,然后调用到forward方法，获取动作概率和状态值，但这里我们只需要动作概率，所以忽略状态值
        dist = Categorical(action_probs)  # 创建一个Categorical分布对象，基于动作概率分布,相当于采样获得了一系列样本
        action = dist.sample()  # Sample an action from the distribution

        memory.states.append(state)  # Store state in memory
        memory.actions.append(action)  # Store action in memory
        memory.logprobs.append(dist.log_prob(action))  # 储存动作的对数概率到内存中,其中dist.log_prob()是Categorical分布对象的方法，用于计算给定动作的对数概率

        return action.item()  # 上面的action是一个张量，使用item()方法将其转换为Python标量并返回
                              # eg: tensor(1) -> 1
                              
    def update(self, memory):
        # Convert memory to tensors
        old_states = torch.stack(memory.states).to(device).detach()  # states原本是一个list，使用torch.stack()将其转换为一个张量，并移动到指定设备，
                                                                     #同时使用detach()方法将其从计算图中分离出来，以避免在更新过程中计算梯度
                                                                     # eg: [tensor([0.1, 0.2]), tensor([0.3, 0.4])] -> tensor([[0.1, 0.2], [0.3, 0.4]])
        old_actions = torch.stack(memory.actions).to(device).detach()  # Convert actions to tensor
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()  # Convert log probabilities to tensor

        # Monte Carlo rewards
        rewards = []  # Initialize rewards list
        discounted_reward = 0  # Initialize discounted reward
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            # zip()函数同时迭代rewards和is_terminals列表，reversed()函数将它们反转，以便从最后一个时间步开始计算折扣奖励
            if is_terminal:  # If the state is terminal
                discounted_reward = 0  # Reset discounted reward
            discounted_reward = reward + (self.gamma * discounted_reward)  # Compute discounted reward
            rewards.insert(0, discounted_reward)  # 在rewards列表的开头(0代表列表的位置)插入计算得到的折扣奖励
                                                    #(这里的是前面定义的空的list,区别于前边迭代的memory下采样得到的rewards序列)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 将奖励列表转换为指定数据类型的PyTorch张量，并移动到设备上
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 这里的奖励列表储存的已经是MCreturns了
        #对其进行标准化处理，1e-7防止除0,以提高训练的稳定性和效率

        # k次循环后更新策略网络
        for _ in range(self.K_epochs):#_是一个占位符变量，表示我们不关心循环变量的值，只需要循环执行K_epochs次
            # Get action probabilities and state values
            action_probs, state_values = self.policy(old_states)  # 获取当前策略网络对于旧状态的动作概率和状态值
            dist = Categorical(action_probs)  # Create a categorical distribution
            new_logprobs = dist.log_prob(old_actions)  # 计算当前策略网络对于旧动作的对数概率
            entropy = dist.entropy()  # 计算动作分布的熵，熵是一个度量随机性的指标，较高的熵表示动作选择更随机，较低的熵表示动作选择更确定性

            # Calculate ratios
            ratios = torch.exp(new_logprobs - old_logprobs.detach())  #计算概率比率，new_logprobs是当前策略网络对于旧动作的对数概率，
                                                                      # old_logprobs是旧策略网络对于旧动作的对数概率，使用detach()方法将其从计算图中分离出来，以避免在更新过程中计算梯度

            # Advantages
            advantages = rewards - state_values.detach().squeeze()  # 计算优势函数，rewards是之前计算MCreturn，state_values是当前策略网络对于旧状态的状态值，
                                                                    # squeeze()方法用于去除多余的维度，使得state_values的形状与rewards匹配
            # 计算各个不同的loss
            # 计算Actor loss
            # 计算surrogate loss，PPO的核心思想是通过剪切概率比率来限制策略更新的幅度，以保持训练的稳定性。
            # surr1是原始的优势乘以概率比率，surr2是剪切后的优势乘以剪切后的概率比率。最终的损失函数取两者的最小值，以确保更新不会过大。
            surr1 = ratios * advantages  # Surrogate loss 1
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 裁切限制surr2的大小，防止r_t过大导致策略更新过大
            loss_actor = -torch.min(surr1, surr2).mean()  # Actor loss,取两者的最小值，并取负号，
                                                          # 因为我们希望最大化优势函数，而优化器默认是最小化损失函数，所以需要取负号来反转优化目标

            # 计算Critic loss
            loss_critic = self.MseLoss(state_values.squeeze(), rewards)  # Critic loss

            # 计算Total loss
            loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy.mean()  # Combined loss
            # actor和critic以及entropy的损失的加权平均,actor占主导，critic占次要，entropy作为正则化项鼓励探索，权重较小
            
            # 记录loss的历史，以便后续分析和可视化
            self.loss_history.append(loss.item())
            self.actor_loss_history.append(loss_actor.item())
            self.critic_loss_history.append(loss_critic.item())
            
            # Update policy
            self.optimizer.zero_grad()  # 清空优化器的梯度缓存，以准备进行新的反向传播
            loss.backward()  # 根据计算图反向传播计算损失函数的梯度，更新模型参数的梯度信息
            self.optimizer.step()  # 执行优化器的更新步骤，根据计算得到的梯度更新模型参数

        # 更新旧策略网络的参数，k_epochs次循环后更新一次,使其与当前策略网络保持一致，以便在下一次选择动作时使用最新的策略
        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy new policy parameters to old policy

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
env = gym.make("CartPole-v1")  # Initialize CartPole environment
state_dim = env.observation_space.shape[0]  # env.observation_space是状态空间对象,加了.shape[0]是为了获取状态空间的维度,即状态向量的长度
action_dim = env.action_space.n  # env.action_space是动作空间对象,加了.n是为了获取动作空间的维度,即可用动作的数量
lr = 0.002  # Learning rate
gamma = 0.99  # Discount factor
eps_clip = 0.2  # Clipping parameter
K_epochs = 4  # Number of epochs for policy update
max_episodes = 1000  # Maximum number of episodes
max_timesteps = 300  # Maximum timesteps per episode

# PPO Training
ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)  # Initialize PPO agent
memory = Memory()  # 初始化memory，用于存储代理的经验
reward_history = []

for episode in range(1, max_episodes + 1):  # Loop over episodes
    state, _ = env.reset()  # 重置环境，获取初始状态,env.reset()返回一个元组(state, info)，
                            # 其中state是环境的初始状态，info是一个包含额外信息的字典，这里我们只需要state，所以使用_来忽略info
    total_reward = 0  # Initialize total reward

    # 在每个episode中，agent与环境交互，直到达到最大时间步数或环境返回终止信号
    for t in range(max_timesteps):  # Loop over timesteps
        action = ppo.select_action(state, memory)  # 根据当前状态选择动作，并将相关信息存储到memory中
        state, reward, done, _, _ = env.step(action)  # 在环境中采取行动，获取新的状态、奖励、是否终止等信息,
                                                      # env.step(action)返回一个元组(state, reward, done, truncated, info)，

        memory.rewards.append(reward)  # 存储奖励到memory中,这里的reward是环境返回的即时奖励
        memory.is_terminals.append(done)  # 存储是否终止的标志到memory中,done是一个布尔值，表示当前时间步是否为终止状态
        total_reward += reward  # Accumulate total reward

        if done:  # If episode is done
            break  # Exit loop

    ppo.update(memory)  # Update PPO agent
    memory.clear()  # Clear memory
    
    reward_history.append(total_reward)

    print(f"Episode {episode}, Total Reward: {total_reward}")  # Print episode statistics

# 保存最终模型参数
torch.save(ppo.policy.state_dict(), "ppo_cartpole.pth")
print("Model saved")

# 可视化训练过程中的reward和loss变化
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(reward_history)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1,2,2)
plt.plot(ppo.loss_history)
plt.title("Training Loss")
plt.xlabel("Update Step")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

env.close()  # Close the environment


# Testing the trained model
test_env = gym.make("CartPole-v1", render_mode="human")

test_policy = ActorCritic(state_dim, action_dim).to(device)
test_policy.load_state_dict(torch.load("ppo_cartpole.pth"))
test_policy.eval()   # 切换为测试模式

state, _ = test_env.reset()

for _ in range(1000):

    state_tensor = torch.FloatTensor(state).to(device)

    with torch.no_grad():  # 测试时不计算梯度
        action_probs, _ = test_policy(state_tensor)

    action = torch.argmax(action_probs).item()

    state, reward, done, _, _ = test_env.step(action)

    if done:
        state, _ = test_env.reset()

test_env.close()
