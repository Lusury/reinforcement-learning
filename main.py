
import numpy as np
from scipy.stats import norm
import pandas as pd

# %%

np.set_printoptions(suppress=True)  # 打印实际值，不用科学计数法
# 初始化
T = 10  # 时间范围10年
lnw_min = np.zeros(T + 1)
lnw_max = np.zeros(T + 1)
w = np.zeros(T + 1)
action = np.zeros(T + 1)  # 初始动作, 取值范围[0, 14]
policy = np.zeros(T + 1)
w[0] = 100  # 财富初始值
H = 200  # 最终财富目标
epsilon = 0.3
alpha = 0.1
gamma = 1

mu_min = 0.0526
mu_max = 0.0886
sig_max = 0.0945

lnw_min[0] = np.log(w[0])
lnw_max[0] = np.log(w[0])
for t in range(1, T + 1):
    lnw_min[t] = lnw_min[t - 1] + (mu_min - 0.5 * sig_max * sig_max) - 3 * sig_max
    lnw_max[t] = lnw_max[t - 1] + (mu_max - 0.5 * sig_max * sig_max) + 3 * sig_max
lnw_array = np.linspace(lnw_min, lnw_max, 101)

# 生成所有的财富值，财富值维度是101行，11列，第一列都是初始值100，第一行是每个时间点的最小值，最后一行是最大值
w_array = np.exp(lnw_array)

# %%
K = 15  # 15种资产配置
Q = np.zeros((len(w_array), T + 1, K))  # 状态动作函数，体现了奖励，101指财富W的101种可能结果，对财富做了离散化处理
R = np.zeros((len(w_array), T + 1, K))  # 状态奖励

# 最终回报的计算，最终财富有101种可能，对应有101种回报R，[:]表示和动作无关，这里是状态奖励
for j in range(len(w_array)):
    if w_array[j][T] > H:
        R[j][T][:] = 1

EF_mu = pd.read_excel('portfolios.xlsx')['miu'].values
EF_sig = pd.read_excel('portfolios.xlsx')['sigma'].values


# %%


def state_change(w0, w1, a0):
    """
    状态转移函数，人为设定的环境变化情况
    :param w0: 当前财富值
    :param w1: 下一状态可能的财富值，是个101维向量，离散化操作
    :param a0: 当前选择的动作，15个mu和sig的组合, 取值是[0,14]
    :return: 下一状态的下标
    """
    mu = EF_mu[a0]
    sig = EF_sig[a0]
    p1 = norm.pdf(np.log(w1 / w0) - (mu - 0.5 * sig ** 2) / sig)
    p1 = p1 / sum(p1)
    idx1 = len(np.where(np.random.rand() > p1.cumsum())[0])
    return idx1


def update_Q_function(idx0, t0):
    """
    更新Q函数
    :param idx0: 财富下标
    :param t0: 时间
    :return: 下一状态的下标，时间
    """
    q = Q[idx0, t0, :]
    a_max = np.where(q == q.max())[0]
    if len(a_max) > 1:
        a_max = np.random.choice(a_max)

    # 使用epsilon贪心算法
    if np.random.rand() < epsilon:
        a0 = np.random.randint(0, K)
    else:
        a0 = a_max

    # 生成下一个状态，更新Q函数
    t1 = t0 + 1
    if t0 < T:
        w0 = w_array[idx0][t0]  # 标量
        w1 = w_array[:, t1]  # 向量
        idx1 = state_change(w0, w1, a0)
        Q[idx0, t0, a0] = Q[idx0, t0, a0] + alpha * (R[idx0, t0, a0] + gamma * Q[idx1, t1, :].max() - Q[idx0, t0, a0])
    else:
        Q[idx0, t0, a0] = (1 - alpha) * Q[idx0, t0, a0] + alpha * R[idx0, t0, a0]
        idx1 = idx0
    return idx1, t1, a_max


# %%
for count in range(200000):
    idx = 0
    tt = 0
    for t in range(T + 1):
        idx, tt, policy[t] = update_Q_function(idx, tt)
        print(tt, policy[t])

    print('count:', count)
    print(Q[:, 0, :].max())
    print('============================================')

print(Q)
print('policy:', policy)

