from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity): # capacityサイズのFIFOを生成
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): # メモリにデータを入れる　
        self.memory.append(Transition(*args))

    def sample(self, batch_size): # batch_size分ランダムにメモリから抽出
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)