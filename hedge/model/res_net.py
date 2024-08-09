import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features,dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Sequential()
        
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features)
            )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += self.shortcut(x)  # Add the input to the output
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim, num_layers=2,dropout_rate=0.5):
        super(SimpleResNet, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim,dropout_rate) for _ in range(num_layers)]
        )
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc_input(x)
        out = self.relu(out)
        for block in self.residual_blocks:
            out = block(out)
        out = self.fc_output(out)
        return out

if __name__ == '__main__':
    # 使用示例
    batch_size = 8
    feature_num = 16
    output_dim = 10
    x = torch.randn(batch_size, feature_num)
    model = SimpleResNet(input_dim=feature_num, hidden_dim=32, output_dim=output_dim)
    output = model(x)
    print(output.shape)  # 应该是 (batch_size, output_dim)
