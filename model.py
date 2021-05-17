
from torch import nn
from torch import optim
from torch.nn.modules.activation import Sigmoid


def SimpleMLP():
    return nn.Sequential(
        nn.Linear(22, 484, bias=True),
        nn.Sigmoid(),
        nn.Linear(484, 1, bias=True),
        nn.Sigmoid()
    )


class MLP_model(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(22, 484, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.output2 = nn.Linear(484, 1, bias=True)
        self.sigmoid2 = nn.Sigmoid()
        

    def forward(self, xb):
        return self.lin(xb)

lr = .1
model = SimpleMLP()
opt = optim.SGD(model.parameters(), lr=lr)
epochs = 10

train_dl, test_dl = 
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
