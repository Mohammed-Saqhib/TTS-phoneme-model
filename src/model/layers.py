from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class TTSLayers(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, output_dim):
        super(TTSLayers, self).__init__()
        self.conv1 = ConvLayer(input_dim, 64, kernel_size=5, padding=2)
        self.lstm = LSTMBlock(64, lstm_hidden_dim, num_layers=2, dropout=0.1)
        self.linear = LinearLayer(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(1, 2)  # Change shape for LSTM (batch, seq, feature)
        x = self.lstm(x)
        x = self.linear(x)
        return x