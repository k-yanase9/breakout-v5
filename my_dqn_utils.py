import random
import torch
import math
import gym
import cv2


EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 25000000
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")

def e_greedy_select_action(state,model,device,fire_ball,total_steps):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY) # ランダムアクション選択の閾値を更新
    
    if fire_ball: # ボールが落ちた場合は、強制的にボールを出す
        return torch.tensor([[1]]), eps_threshold
    elif random.random() > eps_threshold:  # モデルによるアクション
        with torch.no_grad():
            return model(state.to(device)).argmax().view(1, 1).cpu(), eps_threshold
    else: # ランダムアクション
        return torch.tensor([[env.action_space.sample()]]), eps_threshold
    
def to_resize_gray(image, resize):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[32:,8:152] # プレイに関係ある部分のみ切り出し
    image = cv2.resize(src=image, dsize=(resize, resize))/255. # 画像のリサイズと値の変換0～1
    for _ in range(10): # ブロックのあるエリアをランダムに削除
        if random.random() > 0.9:
            x_p = random.randint(10, 25)
            y_p = random.randint(0, 70)
            image[x_p:x_p+4, y_p:y_p+10] = 0.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image