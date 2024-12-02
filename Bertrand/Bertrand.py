# %%
import gc
import copy
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# %%
#求解Pn~Pm
PN = 1.472927  #expterm = 8.233506606
PM = 1.924981 
#求解Rn~Rm
RN = 0.2229268029
RM = 0.3374904595

def generate_equidistant_array(min_val, max_val, m):
    if m <= 1:
        return [min_val] * m
    
    step = (max_val - min_val) / (m - 1)
    equidistant_array = [min_val + i * step for i in range(m)]
    
    return equidistant_array

# 定价空间计算
# 空间离散数量
m = 15
PRICING_SPACE = generate_equidistant_array((PN-0.1*(PM-PN)), (PM+0.1*(PM-PN)), m)
# 罚款空间计算
# 空间离散数量
k = 15
PENALTY_SPACE = generate_equidistant_array(0, 2*(RM-RN), k)
print(PRICING_SPACE)
print(PENALTY_SPACE)

# %%
class Pricing_Agent:
    def __init__(self, alpha=0.2, gamma=0.98, beta=0.00001, action_space = PRICING_SPACE):
        """
        代理类，代表一个企业。
        参数:
            alpha (float): 学习率。
            beta (float): epsilon的消散比率
            gamma (float): 未来奖励的折扣因子。
            action_space (list): 定价空间
        """
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.action_space = action_space  # 策略空间
        self.q_matrix = {}      #用字典的形式存储Q-matrix
        for possible_prices in itertools.product(PRICING_SPACE, repeat=2):   
            for possible_penalty in PENALTY_SPACE:    #遍历笛卡尔积，即状态空间是双方的可能价格*罚金 m*m*k
                possible_state = (possible_prices,possible_penalty)
                self.q_matrix[possible_state] = dict((price, 0.0) for price in action_space)       #每个定价（行动）的初始值为0

    def choose_action(self, state, t):
        """
        根据当前状态和时间步选择行动（价格）/选择是explore还是exploit
        参数:
            opponent_price (float): 对手的价格。
            t (int): 当前时间步。
        返回:
            float: 选择的价格。
        """
        epsilon = np.exp(-self.beta * t)
        if np.random.random() < epsilon:
            # explore：随机选择一个价格
            return np.random.choice(self.action_space)
        else:
            return self.get_optimal_action(state)
        
    def get_optimal_action(self,state):
        """
        根据Q-matrix和现在的state选择最优行动
        """
        optimal_actions = [
            action for action, value in self.q_matrix[state].items() if value == max(self.q_matrix[state].values())
            #获取最大qvalue对应的action，若有多个则随机选择
        ]
        return random.choice(optimal_actions)

    def update_q_table(self, state, action, reward, next_state):
        """
        更新Q表。
        参数:
            state (tuple): 双方上一期的价格。
            action (float): 本期的价格决策。
            reward (float): 本期的奖励。
            next_state (tuple): 本期结束双方的价格。
        """
        state_action_value = copy.deepcopy(self.q_matrix[state][action])
        future_next_state_action_value = self.q_matrix[next_state][self.get_optimal_action(next_state)]
        self.q_matrix[state][action] = (1 - self.alpha) * state_action_value + self.alpha * (
                                        reward + self.gamma * future_next_state_action_value)

# %%
class Regulator:
    def __init__(self, alpha=0.2, gamma=0.98, beta=0.00001, action_space = PENALTY_SPACE):
        """
        监管者类，在市场中对算法合谋行为进行自动监管
        参数:
            alpha (float): 学习率。
            beta (float): epsilon的消散比率
            gamma (float): 未来奖励的折扣因子。
            action_space (list): 罚款空间
        """
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.action_space = action_space  # 策略空间
        self.q_matrix = {}      #用字典的形式存储Q-matrix
        for possible_state in itertools.product(PRICING_SPACE, repeat=2):       #遍历笛卡尔积，即状态空间是双方的可能价格 m*m
            self.q_matrix[possible_state] = dict((penalty, 0.0) for penalty in action_space)       #每个罚款（行动）的初始值为0

    def choose_action(self, state, t):
        """
        根据当前状态和时间步选择行动（价格）/选择是explore还是exploit
        参数:
            opponent_price (float): 对手的价格。
            t (int): 当前时间步。
        返回:
            float: 选择的价格。
        """
        epsilon = np.exp(-self.beta * t)
        if np.random.random() < epsilon:
            # explore：随机选择一个价格
            return np.random.choice(self.action_space)
        else:
            return self.get_optimal_action(state)
        
    def get_optimal_action(self,state):
        """
        根据Q-matrix和现在的state选择最优行动
        """
        optimal_actions = [
            action for action, value in self.q_matrix[state].items() if value == max(self.q_matrix[state].values())
            #获取最大qvalue对应的action，若有多个则随机选择
        ]
        return random.choice(optimal_actions)

    def update_q_table(self, state, action, reward, next_state):
        """
        更新Q表。
        参数:
            state (tuple): 本期双方的价格决策。
            action (float): 监管者对合谋收取的罚金。
            reward (float): 本期的消费者剩余。
            next_state (tuple): 下一期双方的定价决策。
        """
        state_action_value = copy.deepcopy(self.q_matrix[state][action])
        future_next_state_action_value = self.q_matrix[next_state][self.get_optimal_action(next_state)]
        self.q_matrix[state][action] = (1 - self.alpha) * state_action_value + self.alpha * (
                                        reward + self.gamma * future_next_state_action_value)

# %%
def calculate_demand(prices, a=2, mu=0.25):
    """
    根据multinomial loGIt模型计算需求。
    参数:
    prices (list): 所有公司的价格列表。
    返回:
    demands (list): 每个公司的需求列表。
    """
    n = len(prices)
    qualities = [a]*n
    exp_terms = np.array([((qualities[i] - prices[i] )/ mu) for i in range(n)])
    denominator = 1 + np.sum(np.exp(exp_terms))
    demands = np.exp(exp_terms) / denominator
    return demands

def calculate_profit(prices,c=1):
    """
    计算每个公司的利润。
    参数:
    prices (list): 所有公司的价格列表。
    返回:
    profits (list): 每个公司的利润列表。
    """  
    costs = [c]*len(prices)
    demands = calculate_demand(prices)
    profits = [(prices[i] - costs[i]) * demands[i] for i in range(len(prices))]
    return profits

def calculate_penalty(penalty,profits,penalty_mode = 0):
    """
    计算每个厂商被收取罚金之后的利润
    参数：
    penalty (float)：regulator输出的罚金，实际上为平均罚金
    profits (list)：每个厂商的原始利润
    mode (int)：罚金模式，mode为1时对各厂商收取等价罚金，mode为2时按照利润进行分配
    """
    if penalty_mode == 0:
        rewards_after_penalty = profits
    elif penalty_mode == 1:    # 收取相等的罚金
        rewards_after_penalty = [x - penalty for x in profits]
    else:    # 按照超额利润的比例收取罚金
        new_list = [x - RN for x in profits]
        sum_list = sum(new_list)
        penalty_by_percentage = [(x / sum_list * penalty * len(new_list)) for x in new_list]
        rewards_after_penalty = [x - y for x, y in zip(profits, penalty_by_percentage)]
    return  rewards_after_penalty

def calculate_consumer_walfare(prices, mu=0.25, a=2):
    """
    根据新的定价计算消费者福利
    参数：
    prices (tuple/list)：每个厂商的定价
    返回:
    walfare (float): 社会福利
    """
    new_list = [np.exp((a-x)/mu) for x in prices]
    sum_list = sum(new_list)
    walfare = mu * np.log(1+sum_list)  # np.log计算的是自然对数
    return walfare

signal_list = []
def generate_signal(input, signal_mode=0, para=0):
    if signal_mode == 0:  
        signal = input
    elif signal_mode == 1: # 在罚金范围内随机生成
        signal = random.choice(PENALTY_SPACE)
    elif signal_mode == 2: # 返回固定的signal
        signal = para
    else: # 根据一定的time—lag，生成signal
        global signal_list
        signal_list.append(input)
        if len(signal_list) >= 6:
            signal = signal_list.pop(0)
        else:
            signal = 0 
    return signal

# %%
class Market:
    def __init__(self, pricing_agents, regulator):
        self.agents = pricing_agents
        self.regulator = regulator

    def simulation(self,end_episode=2000000,penalty_mode=0,signal_mode=0):
        # 初始化
        agents = self.agents
        regulator = self.regulator
        prices_record = []
        penalty_record = []
        signal_record = []
        rewards_record = []
        walfare_record = []
        prices = (PRICING_SPACE[0],PRICING_SPACE[0])
        penalty = 0
        walfare = 0
        signal = 0
        # 开始循环
        for episode in tqdm(range(end_episode)):
            # 定价代理选择定价
            new_prices = [0,0]
            new_penalty = 0
            for i in range(len(agents)):
                agent_action = agents[i].choose_action((prices,signal),episode)
                new_prices[i] = agent_action  # 更新价格
            new_prices = tuple(new_prices)

            # 计算销售利润
            profits = calculate_profit(new_prices)
            # 计算本期的消费者福利
            walfare = calculate_consumer_walfare(new_prices)
            
            # 消费者福利为上一期监管者的奖励，监管者进行学习
            regulator.update_q_table(prices,  # 上期价格
                                    penalty,    # 上期收取的罚金
                                    walfare,    # 本期的消费者福利，是上一期的奖励
                                    new_prices)    # 收取罚金后的定价
         
            # 收取罚金
            new_penalty = regulator.choose_action(new_prices,episode)
            # 计算扣除罚金后的实际利润
            # mode：【0】最终利润中不扣除罚金，即罚金不落地；【1】双方平分penalty；【2】双方根据超额礼利润分配
            rewards = calculate_penalty(new_penalty,profits,penalty_mode=penalty_mode)

            # 根据regulator生成的penalty，生成一个signal给agent
            # mode：【0】signal=penalty；【1】从列表中返回一个随机值；【2】返回一个固定值para；【3】根据一定的time—lag，生成signal
            # [0.0, 0.016366236657142857, 0.032732473314285714, 0.04909870997142857, 0.06546494662857143, 0.08183118328571429, 0.09819741994285713, 
            # 0.1145636566, 0.13092989325714285, 0.1472961299142857, 0.16366236657142857, 0.18002860322857142, 0.19639483988571427, 0.21276107654285714, 0.2291273132]
            new_signal = generate_signal(new_penalty,signal_mode=signal_mode,para=0)
            
            # 定价代理进行学习
            for i in range(len(agents)):
                agents[i].update_q_table((prices,signal),   # 状态
                                        new_prices[i],  # 自己的价格决策
                                        rewards[i],    # 实际获得的利润
                                        (new_prices,new_signal)) # 下一期的状态
                
            # 记录本周期的数据
            prices_record.append(new_prices)
            penalty_record.append(new_penalty)
            signal_record.append(new_signal)
            rewards_record.append(rewards)
            walfare_record.append(walfare)
            prices = new_prices
            penalty = new_penalty
            signal = new_signal
            
        return prices_record,penalty_record,signal_record,rewards_record,walfare_record

# %%
def calculate_average(data, chunk_size=1000):
    if isinstance(data[0], (int, float)):  # 单列
        return [np.mean(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]
    elif isinstance(data[0], (tuple, list)):  # 多列
        num_columns = len(data[0])
        reduced_columns = []
        for col_idx in range(num_columns):
            col_data = [row[col_idx] for row in data]
            reduced_columns.append(
                [np.mean(col_data[i:i + chunk_size]) for i in range(0, len(col_data), chunk_size)]
            )
        return list(zip(*reduced_columns))

def save_data(all_lists):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./data/{timestamp}.xlsx"
    file_name

    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
        for index in range(len(all_lists)):
            average_data = calculate_average(all_lists[index], chunk_size=1000)
            if isinstance(average_data[0], (int, float)):  # 单列
                df = pd.DataFrame({'average': average_data})
            else:  # 多列
                num_columns = len(average_data[0])
                column_names = [f"Column{i+1}_avg" for i in range(num_columns)]
                df = pd.DataFrame(average_data, columns=column_names)
            df.to_excel(writer, index=False, sheet_name=f"sheet_{index}")
    # 释放内存
    del all_lists
    gc.collect()

# %%
def recurrent_simulation(simulation_times,episodes=2000000,penalty_mode=0,signal_mode=0):
    for i in range(simulation_times):
        agent1 = Pricing_Agent()
        agent2 = Pricing_Agent()
        regulator = Regulator()

        market = Market([agent1,agent2],regulator)
        prices_record,penalty_record,signal_record,rewards_record,walfare_record = market.simulation(end_episode=episodes,penalty_mode=penalty_mode,signal_mode=signal_mode)

        save_data([prices_record,penalty_record,signal_record,rewards_record,walfare_record])
        
    print(f"simulation_{i+1} completed!")

# %%
if __name__ == "__main__":
    recurrent_simulation(20,episodes=2000000,penalty_mode=0,signal_mode=0)


