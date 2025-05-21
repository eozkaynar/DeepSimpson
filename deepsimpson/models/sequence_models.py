import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=1):
        """
        input_size: The number of features in the input at each time step. 
        hidden_size: The number of features in the hidden state.
        num_layers: The number of recurrent layers in the network.
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1         = nn.RNN(input_size, hidden_size, batch_first = True)
        self.rnn2         = nn.RNN(128, 32, batch_first = True)
        self.rnn3         = nn.RNN(32, 16, batch_first = True)

        # x needs to be: (batch_size, seq, input_size)

        # Regression
        self.fc          = nn.Linear(16, num_classes)

    def forward(self, x):
        #  Set initial hidden states
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn1(x, h0)

        # RNN2 için init state
        h1 = torch.zeros(1, x.size(0), 32, device=x.device)
        out, _ = self.rnn2(out, h1)

        # RNN3 için init state
        h2 = torch.zeros(1, x.size(0), 16, device=x.device)
        out, _ = self.rnn3(out, h2)
        out    = out[:,-1,:]     # Only last time step

        out    = self.fc(out)

        return out  
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=1):
        """
        input_size: The number of features in the input at each time step. 
        hidden_size: The number of features in the hidden state.
        num_layers: The number of recurrent layers in the network.
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1         = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.lstm2         = nn.LSTM(hidden_size, 32, batch_first = True)
        self.lstm3         = nn.LSTM(32, 16, batch_first = True)

        # x needs to be: (batch_size, seq, input_size)

        # Regression
        self.fc          = nn.Linear(16, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)  # No need to pass initial hidden state
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        
        # Use the last time step's output
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        out = self.fc(out)
        return out
