import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialLSTM(nn.Module):

    def __init__(self, hidden_size, grid_range, num_grids):
        '''
            Initialize the Social LSTM
            grid_range: the minimum and maximum of the grid the coordinates lies in, would be [(x_min, x_max), (y_min, y_max)]
            num_grids, number of grids on each diension, if it is n, then there will be n ^ 2 grids in total
        '''
        super(SocialLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_grids = num_grids
        self.grid_range = grid_range
        # define social LSTM layers
        self.lstm = nn.LSTMCell(3, hidden_size)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def social_layer(self, old_hidden_state, coords):
        '''
            Given an old hidden state of shape (n, hs) and coordinates of shape (n, 2),
            Return a new hidden state of shape (n, hs)
        '''
        coords = coords.clone().to(self.device)
        coords[:, 0] = coords[:, 0].clamp(*self.grid_range[0])
        coords[:, 1] = coords[:, 1].clamp(*self.grid_range[1])
        # First make a dictionary to store all possible grids
        total_length_x = self.grid_range[0][1] - self.grid_range[0][0]
        total_length_y = self.grid_range[1][1] - self.grid_range[1][0]
        x_min, y_min = self.grid_range[0][0], self.grid_range[1][0]
        dx, dy = total_length_x / self.num_grids, total_length_y / self.num_grids

        # create a social dict
        social_dict = [[
         [ torch.zeros(1, self.hidden_size).to(self.device), []]
         for j in range(self.num_grids)   
        ] for j in range(self.num_grids)]

        # helper function for locating the grid
        def get_grid_by_coordinates(c):
            x, y, _ = c.cpu().clone().numpy()

            x_ind = min(max(int((x - x_min) // dx), 0), self.num_grids - 1)
            y_ind = min(max(int((y - y_min) // dy), 0), self.num_grids - 1)
            try:
                return social_dict[x_ind][y_ind]
            except IndexError:
                breakpoint()
                return None

        # now iterate each "batch" (node in this particular timestamp), and accumulate the corresponding state at the social_dict
        for idx, hidden_state in enumerate(torch.split(old_hidden_state, 1, dim = 0)):
            grid = get_grid_by_coordinates(coords[idx])
            
            if not grid: 
                # if there are no grids found...
                raise ValueError("Failed to find any grids for the coordinate: {}".format(coords[idx]))

            # accumulate hidden state
            grid[0] += old_hidden_state[idx]
            # mark down the idx used here
            grid[1].append(idx)

        # now distribute back
        new_hidden_state = old_hidden_state.clone()
        for i in range(self.num_grids):
            for j in range(self.num_grids):
                social_hidden_state, ids = social_dict[i][j]
                new_hidden_state[ids] = social_hidden_state
        
        return new_hidden_state

    def forward(self, coords, hidden_state, cell_state):
        '''
            Forward one step of coordinates of size (num_batch, 2) into this Social LSTM
            return the hidden state and cell state after it is passed
        '''
        # initial states
        # hidden_state, cell_state = torch.zeros(num_batch, self.hidden_size), torch.zeros(num_batch, self.hidden_size)
        hidden_state, cell_state = self.lstm(coords, (hidden_state, cell_state))
        # social layer
        hidden_state = self.social_layer(hidden_state, coords)

        return hidden_state, cell_state

class SocialModel(nn.Module):
    
    def __init__(self):
        super(SocialModel, self).__init__()

        self.hidden_size = 128
        self.linear_intermediate_size = 64

        self.slstm = SocialLSTM(self.hidden_size, [(-1, 1), (-1, 1)], 7)

        self.linear1 = nn.Linear(self.hidden_size, self.linear_intermediate_size)
        self.linear2 = nn.Linear(self.linear_intermediate_size, 5)

    def zero_initial_state(self, num_nodes):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.zeros((num_nodes, self.hidden_size)).to(device), torch.zeros((num_nodes, self.hidden_size)).to(device)
    
    def forward(self, x, initial_states = None):
        '''
            TODO: this
            Given input tensor of shape (batch_size, seq_length, 2), 
            output values of size 5 
        '''
        num_nodes = x.size(0)
        num_steps = x.size(1)

        if initial_states:
            hs, cs = initial_states
        else:
            hs, cs = self.zero_initial_state(num_nodes)

        list_hs = []
        for ts in range(num_steps):
            # the hs here should be "socialized" already
            hs, cs = self.slstm(x[:, ts, ...], hs, cs)
            list_hs.append(hs)
        list_hs = torch.stack(list_hs)
        # linear inference
        out = torch.relu(self.linear1(list_hs))
        out = self.linear2(out)

        return out.transpose(1, 0), hs, cs

class VanillaLSTMModel(nn.Module):
    
    def __init__(self, hidden_size, intermediate_hidden_size):
        super(VanillaLSTMModel, self).__init__()

        self.lstm = nn.LSTMCell(3, hidden_size)

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(hidden_size, intermediate_hidden_size)
        self.linear2 = nn.Linear(intermediate_hidden_size, 5)
    
    def zero_initial_state(self, num_nodes):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.zeros((num_nodes, self.hidden_size)).to(device), torch.zeros((num_nodes, self.hidden_size)).to(device)
    
    def forward(self, x, initial_states = None):
        num_nodes = x.size(0)
        num_steps = x.size(1)

        if initial_states:
            hs, cs = initial_states
        else:
            hs, cs = self.zero_initial_state(num_nodes)

        list_hs = []
        for ts in range(num_steps):
            # the hs here should be "socialized" already
            hs, cs = self.lstm(x[:, ts, ...], (hs, cs))
            list_hs.append(hs)
        list_hs = torch.stack(list_hs)
        # linear inference
        out = torch.relu(self.linear1(list_hs))
        out = self.linear2(out)

        return out.transpose(1, 0), hs, cs
if __name__ == '__main__':
    import torch
    # from loss import Gaussian2DLoss
    from torch.optim import SGD, Adam
    import torch.nn.functional as F
    grid_range = [(-1, 1), (-1, 1)]
    num_grids = 7
    num_steps = 8
    batch_size = 20
    hidden_size = 128
    dummy_dataset = torch.randn(batch_size, num_steps, 2).clamp(-1, 1)

    slstm = SocialLSTM(hidden_size, grid_range, num_grids)

    optimizer = Adam(slstm.parameters(), 1e-3)
    # show number of parameters for optimization
    print('There are {} parameters'.format(sum([module.numel() for module in slstm.parameters()])))
    # try training on this to test if the autograd graph is connected in between
    for epoch in range(100): # for each epoch..
        optimizer.zero_grad()
        # initial state
        hs, cs = torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size)
        # forward pass
        total_loss = 0
        for ts in range(num_steps):
            hs, cs = slstm(dummy_dataset[:, ts, ...], hs, cs)
            loss = hs.norm() # dummy loss
            total_loss += loss.item()
            loss.backward(retain_graph = True)
        optimizer.step()
        print('Epoch: {}, loss: {}'.format(epoch, total_loss / num_steps))