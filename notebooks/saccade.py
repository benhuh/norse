import torch
from torch import nn, jit
from dyn_weight import Linear_Active
import numpy as np

class RNNCell_base(jit.ScriptModule):     # (nn.Module):   # (RNNCell_base):  # Euler integration of rate-neuron network dynamics 
    def __init__(self, input_size, hidden_size, output_size, nonlinearity = None, decay = 0.9, bias = True):
        super().__init__()
    
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.decay = decay    #  torch.exp( - dt/tau )

    def initialize_hidden(self, batch):
        hidden_init = torch.zeros(batch, self.hidden_size) 
        return hidden_init
    
    def hidden_dynamics(self, hidden, activity):
        if self.nonlinearity is not None:
            activity = self.nonlinearity(activity)
        hidden   = self.decay * hidden + (1 - self.decay) * activity
        return hidden
    
        
class RNNCell_active(RNNCell_base):     # (nn.Module):   # (RNNCell_base):  # Euler integration of rate-neuron network dynamics 
    def __init__(self, input_size, hidden_size, output_size, nonlinearity = None, decay = 0.9, bias = True):
        super().__init__(input_size, hidden_size, output_size, nonlinearity, decay, bias)
        
        n_context = 1
        self.in_layer  = Linear_Active(input_size, hidden_size, n_context,  gain_w = 0.01, gain_b = 0.01, passive = True)
        self.rec_layer = Linear_Active(hidden_size, hidden_size, n_context, gain_w = 0.01, gain_b = 0.01, passive = True)
        self.out_layer = Linear_Active(hidden_size, output_size, n_context, gain_w = 0.01, gain_b = 0.01, passive = True)
        
        
    def forward(self, input, hidden, context):        
        activity = self.in_layer(input, context) + self.rec_layer(hidden, context) 
        hidden   = self.hidden_dynamics(hidden, activity)
        output   = self.out_layer(hidden, context)
        return hidden, output

           
class RNNCell_passive(RNNCell_base):     # (nn.Module):   # (RNNCell_base):  # Euler integration of rate-neuron network dynamics 
    def __init__(self, input_size, hidden_size, output_size, nonlinearity = None, decay = 0.9, bias = True):
        super().__init__(input_size, hidden_size, output_size, nonlinearity, decay, bias)
        
        n_context = 1
        self.in_layer  = nn.Linear(input_size, hidden_size, bias = bias)
        self.rec_layer = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size, bias = bias)
        
#         self.out_layer.weight = Parameter(torch.zeros_like(self.out_layer.weight))
#         self.out_layer.bias = Parameter(torch.zeros_like(self.out_layer.bias))
        
    def forward(self, input, hidden, context):
        activity = self.in_layer(input) + self.rec_layer(hidden) 
        hidden   = self.hidden_dynamics(hidden, activity)
        output   = self.out_layer(hidden)
        return hidden, output
    
    
class Plant(nn.Module):
    def __init__(self, tau, decay):
        super().__init__()
        self.pos_decay = decay # 0.95
        self.vel_decay = decay # 0.95
        self.tau = tau

    def initialize_state(self, batch, seq_length, amp = 8, amp0 = 0.1):
        target_seq = Gen_target_seq(batch, seq_length, amp)
        pos = torch.from_numpy(target_seq[0]).float() + amp0 * (torch.rand(batch, 2) - 0.5)
        vel = amp0 / self.tau * (torch.rand(batch, 2)  - 0.5)
        state_init = torch.cat([pos, vel], dim=1) 
        return state_init, target_seq
    
    def forward(self, state, force):
        pos, vel = state.split(2, dim=1) 
        acc = force - (1 - self.pos_decay) * pos / self.tau ** 2 - (1 - self.vel_decay) * vel / self.tau
        vel += acc   # mass = 1, dt = 1
        pos += vel 
        state = torch.cat([pos, vel], dim=1) #
        return state, acc
    

class RNN_Plant(nn.Module):    
    def __init__(self, n_rnn, tau, decay,  nonlinearity = None, rnn_type = 'active'):
        super().__init__()
        n_input = 2 + 2 + 2 + 1 # target, pos, vel, go_cue
        n_output = 2 
        self.n_rnn = n_rnn
        self.tau = tau
        self.decay = decay
        self.rnn_type = rnn_type
        
        self.loss_fnc = state_action_loss
        self.plant  = Plant(tau, decay)
        
        if rnn_type == 'active':
            self.rnn    = RNNCell_active(n_input, n_rnn, n_output, nonlinearity = nonlinearity, decay = 0.9, bias = True )
        else:
            self.rnn    = RNNCell_passive(n_input, n_rnn, n_output, nonlinearity = nonlinearity, decay = 0.9, bias = True )
        

    def forward(self, batch, seq_length): 
        # Initialize
        state_init, target_seq  = self.plant.initialize_state( batch, seq_length, amp = 8, amp0 = 0.1)
        hidden_init = self.rnn.initialize_hidden(batch) 
        
        state, action, hidden, go, target, target_go, mask = self.simulate_loop(state_init, hidden_init, target_seq, seq_length)
        loss = self.calculate_loss(state, action, hidden, target_go, go, mask)
        sim_result = state.detach().cpu().numpy(), hidden.detach().cpu().numpy(), go.detach().cpu().numpy(), target.detach().cpu().numpy(), target_go.detach().cpu().numpy()
        return loss, sim_result
    
            
    def simulate_loop(self, state, hidden, target_seq, seq_length):

        batch = state.shape[0]
        time_points, mask = generate_time_points(batch, self.tau, seq_length-1)
        t_max = time_points[-1]
        
        go_all, target_all, target_go_all, hidden_all, action_all, state_all = [], [], [], [], [] ,[]
        
        for i in range(t_max):  # looping over the time dimension 
            
            go, target, target_go = generate_go_target(i, time_points, target_seq)
            
            scale = torch.tensor([1/4, 1/4, 1/4, 1/4, 1/4, 4*self.tau, 4*self.tau]).view(1,-1) 
            input = scale * torch.cat([go, target, state], dim=1)  
            
            hidden, action = self.rnn(input, hidden, go)  #  action = self.readout(hidden)
            state, acc  = self.plant(state, action)
            
            go_all += [go]  
            target_all += [target]  
            target_go_all += [target_go]  
            hidden_all += [hidden]  
            action_all += [acc]  #+= [action]  
            state_all  += [state]  

        return torch.stack(state_all), torch.stack(action_all), torch.stack(hidden_all), torch.stack(go_all), torch.stack(target_all), torch.stack(target_go_all), mask 

    def calculate_loss(self, state, action, hidden, target, go, mask):
        loss     = self.loss_fnc(state, action, target, go, self.tau, mask) #, output)
        
## regularization cost: activity / weights
#         loss += (hidden.abs().mean() * hp['l1_h'] + hidden.norm() * hp['l2_h'])  #    Regularization cost  (L1 and L2 cost) on hidden activity
#         for param in self.parameters():
#             loss += param.abs().mean() * hp['l1_weight'] + param.norm() * hp['l2_weight']   #    Regularization cost  (L1 and L2 cost) on weights
        return loss



def state_action_loss(state, acc, target, go, tau, mask):
    pos, vel = state.split(2, dim=2)
    loss = (mask * (1 - (- (pos - target) ** 2 - 2 * (vel*tau) ** 2).exp() + (acc * tau**2) ** 2)).mean()
    loss += (mask * 10 * (go * ((vel*tau) ** 2).mean(dim=2, keepdim=True))).mean()
    return loss

def Gen_target_seq(batch, seq_length, amp):
    random_targets = amp * ( np.random.rand(seq_length, batch, 2) - 0.5)  # 2D targets
    return random_targets
    
def generate_time_points(batch, tau, seq_length = 2):
    
    t0 = -5
    t_on_all, t_jump_all, t_off_all = [], [], []
    for i in range(seq_length):
        t_on = t0 + 15 + 5*np.random.rand(batch)  # go_on ... fixation
        t_jump = t_on + 5 + 10*np.random.rand(batch)  # target_jump while fixation
        t_off =  t_jump + 5 + 15*np.random.rand(batch) # delay period -> go
        
        t0 = t_off        
        t_on_all += [t_on*tau];    t_jump_all += [t_jump*tau];    t_off_all += [t_off*tau]

    t_max = int((t_off.max() + 15) *tau )
    
    t_all = np.arange(t_max)
    mask = np.ones((t_max, 1, 1))  
    mask[t_all<10] *= 0
    mask = torch.from_numpy(mask).float() 
    
    time_points = t_on_all, t_jump_all, t_off_all, t_max
    return time_points, mask

def generate_go_target(i, time_points, target_seq):
    t_on, t_jump, t_off, t_max = time_points
    
    go = np.zeros((target_seq[0].shape[0],1))
    target = np.copy(target_seq[0])
    target_go = np.copy(target_seq[0])
    
    
    for k in range(len(t_on)):
        bool_go     = np.logical_and(t_on[k]<i, i<t_off[k])
        bool_target = t_jump[k] < i
        bool_off    = t_off[k] < i
        
        go[bool_go] += 1.0
        target[bool_target] = np.copy(target_seq[k+1,bool_target])
        target_go[bool_off] = np.copy(target_seq[k+1,bool_off])            

    go        = torch.from_numpy(go).float().view(-1,1)
    target    = torch.from_numpy(target).float()
    target_go = torch.from_numpy(target_go).float()

    return go, target, target_go