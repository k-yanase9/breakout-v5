import random
import math

import gymnasium as gym

from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from my_logger import my_log
from my_model import Dueling_Network
from my_memory import ReplayMemory, Transition
from my_dqn_utils import e_greedy_select_action, to_resize_gray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_N_FRAMES = 4 # 過去フレームの利用数
_IMSIZE = 84 # リサイズ後のピクセル数
BATCH_SIZE = 256 # バッチサイズ

# メモリパラメータ
_MEMSIZE = 50000 # メモリのサイズ

# ε-greedy方策パラメータ
GAMMA = 0.99

# モデル更新パラメータ
TAU = 0.005
LR = 1e-4
_NUM_EPISODE_SAVE = 25000 # 何エピソードでモデルを保存するか

# モデルの初期化
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
n_actions = env.action_space.n
policy_net = Dueling_Network(_N_FRAMES, n_actions).to(device)
target_net = Dueling_Network(_N_FRAMES, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

#各種初期値セット
terminated = True
frame1 = True
total_steps = 0
count_update = 0

interval_scores = []
losses = []
m_score = 0
ave_score = 0.0
ave_loss = 0.0
memory = ReplayMemory(capacity=_MEMSIZE)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1000)
    optimizer.step()
    return loss.item()


num_episodes = 2000000 # 学習させるエピソード数
    
for i_episode in range(num_episodes):

    fire_ball = True 
    reward_frame = torch.tensor([0], dtype=torch.float32)  
    if terminated == True:
        state_frame = torch.zeros((1, _N_FRAMES, _IMSIZE, _IMSIZE), dtype=torch.float32)
        next_state_frame = torch.zeros((1, _N_FRAMES, _IMSIZE, _IMSIZE), dtype=torch.float32)
        state, info = env.reset(seed=random.randint(0, 2**24))
        state = to_resize_gray(state, _IMSIZE)
        state_frame[:,0,:,:] = state
        next_state_frame[:,0,:,:] = state
        old_life = info['lives']
    
    for t in count():
        total_steps +=1
        action, eps_threshold = e_greedy_select_action(state_frame, policy_net, device, fire_ball,total_steps)
        fire_ball = False
        observation, reward, terminated, truncated, info = env.step(action.item())
        
        if old_life > info['lives']:
            old_life = info['lives']
            truncated = True
        
        done = terminated or truncated
        if done: # ライフが減った場合にマイナスの報酬を与える
            reward = -1 
            
        reward = torch.tensor([reward])
        reward = torch.clamp(input=reward, min=-1, max=1)
        
        next_state = to_resize_gray(observation, _IMSIZE)
        next_state_frame = torch.roll(input=next_state_frame, shifts=1, dims=1)
        next_state_frame[:,0,:,:] = next_state
            
        if frame1 == True:
            state_frame1 = state_frame
            action_frame1 = action
            next_state_frame1 = next_state_frame
            if done:
                next_state_frame1 = None
            frame1 = False
            
        reward_frame += reward
            
        if (total_steps % _N_FRAMES == 0) or done: 
            memory.push(state_frame1, action_frame1, next_state_frame1, reward_frame)
            frame1 = True    
                 
            count_update += 1
            if count_update % 4 == 0:
                if count_update > _MEMSIZE:
                    losses.append(optimize_model())
            if count_update % 400 == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)   
                target_net.load_state_dict(target_net_state_dict)
                
        state_frame = next_state_frame
        if done:
            break
        
    interval_scores.append(reward_frame.item())
    if i_episode % 5 ==0:
        if len(interval_scores) > 0:
            m_score = max(interval_scores)
            ave_score = sum(interval_scores)/len(interval_scores)
        else:
            m_score = 0
            ave_score = 0
        if len(losses) > 0:
            ave_loss = sum(losses)/len(losses)    
        output = f"i_ep:{i_episode}\tmax_score:{m_score}\tave_score:{ave_score:.3f}\tloss:{ave_loss:.6f}\teps_threshold:{eps_threshold:.6f}"
        print(output+"\r", end="")
    
    if i_episode % 1000 == 0:
        my_log.info(output)
        interval_scores = []
        losses = []

    if (i_episode % _NUM_EPISODE_SAVE) == 0:
        torch.save(target_net.state_dict(), f"./model/{str(i_episode)}.pth")