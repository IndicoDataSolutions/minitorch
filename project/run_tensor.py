"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
from graph_builder import GraphBuilder

import networkx as nx
from pprint import pprint

def RParam(*shape, name=None):
    r = 2 * (minitorch.rand(shape) - 0.5)
    if len(shape) == 1:
        r = minitorch.rand(shape) + 0.1
    return minitorch.Parameter(r, name=name)


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
        layer3_out = self.layer3.forward(x)
        layer3_out.name = "layer_3"
        sigmoid_out = layer3_out.sigmoid()
        sigmoid_out.name = "sigmoid_out"
        return sigmoid_out


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size, name="weight")
        self.bias = RParam(out_size, name="bias")
        self.out_size = out_size

    def forward(self, x):
        # ASSIGN2.5
        batch, in_size = x.shape
        weights_view = self.weights.value.view(1, in_size, self.out_size)
        weights_view.name = "weights_view"
        x_view = x.view(batch, in_size, 1)
        x_view.name = "x_view"
        Wx = weights_view * x_view 
        Wx.name = "Wx"
        sum_Wx = Wx.sum(1)
        sum_Wx.name = "sum_Wx"
        sum_Wx_view = sum_Wx.view(batch, self.out_size)
        sum_Wx_view.name = "sum_Wx_view"
        bias_view = self.bias.value.view(self.out_size)
        bias_view.name = "bias_view"
        layer_out = sum_Wx_view + bias_view
        layer_out.name = "layer_out"
        return layer_out
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

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn, visualize=False):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)
        X.name = "X"
        y.name = "y"

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            out.name = "out"
            prob = (out * y)  + ( - out + 1.0) * (- y + 1.0)
            prob.name = "prob"

            
#            pprint(list(zip(out.to_numpy().tolist(), y.to_numpy().tolist(), prob.to_numpy().tolist())))
            loss = -prob.log()
            loss.name = "loss"
            # l2_pen = (self.model.layer3.weights.value * self.model.layer3.weights.value).sum()
            # print(f"Loss = {loss.sum().view(1)} l2 = {l2_pen}")
            tot_loss = (loss / data.N).sum().view(1) # + 0.000001 * l2_pen.view(1)
            tot_loss.name = "tot_loss"
            graph_builder = GraphBuilder()
            G = graph_builder.run(tot_loss)
#            output_graphviz_svg = nx.nx_pydot.to_pydot(G).create_svg()
#            with open('graph.svg', 'wb') as fd:
#                fd.write(output_graphviz_svg)
 
            print("Total loss", tot_loss, type(tot_loss))
            tot_loss.backward()
            
            if True:
                for param in self.model.parameters():
                    #if (param.value.derivative.to_numpy() > 0).any():
                    print("Epoch", epoch, param.value.derivative.to_numpy())
                    #                    raise AssertionError()
                    print(f"Value: {param.value.to_numpy()},\n Grad:{param.value.derivative.to_numpy()}\n")
                print("-" * 50)
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.get_data() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.005
    data = minitorch.datasets["Diag"](PTS)
    print(data)
    TensorTrain(HIDDEN).train(data, RATE, max_epochs=100000, visualize=True)
