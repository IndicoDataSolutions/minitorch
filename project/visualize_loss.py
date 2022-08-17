"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import minitorch
from minitorch import operators
import minitorch as mt
from pprint import pprint
from matplotlib import pyplot as plt
import pandas as pd

def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


def ones(shape, backend):
    return mt.Tensor.make(
        [1] * int(operators.prod(shape)), shape, backend=backend
    )
    
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        # self.layer1 = Linear(2, hidden_layers)
        # self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(2, 1)

    def forward(self, x):
        # ASSIGN2.5
        # h = self.layer1.forward(x)#.relu()
        #h = self.layer2.forward(h)#.relu()
        return self.layer3.forward(x).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # ASSIGN2.5
        batch, in_size = x.shape
        weights = ones((2, 1), backend=self.bias.value.backend)
        return (
            weights.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)
        # END ASSIGN2.5


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)
        
        records = []
        losses = []
        for epoch in range(1, self.max_epochs + 1):
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + ( - out + 1.0) * (- y + 1.0)
            loss = -prob.log()
            tot_loss = (loss / data.N).sum().view(1) # + 0.000001 * l2_pen.view(1)
            tot_loss.backward()
            total_loss = tot_loss.to_numpy()[0]
            records.append({'bias': self.model.layer3.bias.value, 'grad': self.model.layer3.bias.value.derivative.to_numpy()[0], 'loss': total_loss, 'step': epoch})
            print(records[-1])
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.get_data() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)

        df = pd.DataFrame.from_records(records)
        df.plot(x='step', y='loss')
        plt.savefig('loss-vs-step.png')
        plt.clf()
        df.plot(x='bias', y='loss')
        plt.savefig('loss-vs-bias.png')
        plt.clf()
        df.plot(x='step', y='grad')
        plt.savefig('grad-vs-step.png')
        
if __name__ == "__main__":
    PTS = 200
    HIDDEN = 2
    RATE = 0.005
    data = minitorch.datasets["Diag"](PTS)
    TensorTrain(HIDDEN).train(data, RATE, max_epochs=1000)
