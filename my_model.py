import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Dueling_Network(nn.Module):
    def __init__(self, n_frame, n_actions):
        super(Dueling_Network, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_frame,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.flatten = nn.Flatten()  
        latent_num = self.calc_latentnum(torch.zeros((1,n_frame,84,84)))
        self.act_fc = nn.Linear(latent_num , 512)
        self.act_fc2 = nn.Linear(512, n_actions)
        self.value_fc = nn.Linear(latent_num , 512)
        self.value_fc2 = nn.Linear(512, 1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.act_fc.weight)
        torch.nn.init.kaiming_normal_(self.act_fc2.weight)
        torch.nn.init.kaiming_normal_(self.value_fc.weight)
        torch.nn.init.kaiming_normal_(self.value_fc2.weight)      
        

    def calc_latentnum(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.flatten(x).shape[1]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x_act = self.relu(self.act_fc(x))
        x_act = self.act_fc2(x_act)
        x_val = self.relu(self.value_fc(x))
        x_val = self.value_fc2(x_val)
        x_act_ave = torch.mean(x_act, dim=1, keepdim=True)
        q = x_val + x_act - x_act_ave
        return q
    
if __name__ == "__main__":
    x = torch.zeros((1,6,84,84))
    net = Dueling_Network(6,4)
    print(net.calc_latentnum(x))
    pass