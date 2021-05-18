
import torch
from torch import nn
from torch import optim
from torch.nn.modules.activation import Sigmoid
import preprocess as pp



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

def accuracy(pred, yb):
    """
        determines model prediction accuracy
        uses simple >/< .5 to predict 0 or 1
        returns % correct
    """
    num_correct = 0
    assert len(pred) == len(yb)




def trainMLP():
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.L1Loss()
    lr = .1
    model = SimpleMLP()
    train_dl, test_dl = pp.LoadPreProcess()
    opt = optim.SGD(model.parameters(), lr=lr)
    epochs = 1000

    for epoch in range(epochs):
        ## TRAINING
        model.train()
        for xb, yb in train_dl:
            yb = yb.reshape(len(yb), 1)
            # print(xb.float())
            # print(type(xb))
            pred = model(xb.float())
            # print(len(pred))
            # print(type(pred))
            # print(len(pred[0]))
            assert len(pred) == len(yb)
            # print(pred)
            # print(yb)
            loss = loss_func(pred, yb.long())

            loss.backward()
            opt.step()
            opt.zero_grad()
        ## VALIDATION AND STATS
        model.eval()
        val_loss = 0
        last_val_loss = 0
        num = 0
        num_correct = 0
        num_pay = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                yb = yb.reshape(len(yb), 1)
                # pred = pred.reshape(len(pred), 0)
                pred = model(xb.float())
                assert len(pred) == len(yb)
                val_loss += loss_func(pred, yb.long())
                pred[pred<.5]=0
                pred[pred>=.5]=1
                num_pay += yb.sum()
                num_correct += (pred==yb).sum()
                num += len(yb)

        print('Loss: ', epoch, val_loss)
        print('Successes: ', num_correct)
        print('Accuraccy: ,', num_correct / num)
        print('Num Pay: ', num_pay)
        if last_val_loss < val_loss: print('OVERTRAINING')
        last_val_loss = val_loss


    torch.save(model, './outputs/model.pth')






trainMLP()

# loss_func = nn.CrossEntropyLoss
# lr = .1
# model = SimpleMLP()
# opt = optim.SGD(model.parameters(), lr=lr)
# epochs = 10

# train_dl, test_dl = 
# for epoch in range(epochs):
#     model.train()
#     for xb, yb in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)

#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#     model.eval()
#     with torch.no_grad():
#         valid_loss = sum(loss_func(model(xb), yb) for xb, yb in test_dl)

#     print(epoch, valid_loss / len(test_dl))
