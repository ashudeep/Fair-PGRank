import torch
from torch import nn


class LinearModel(nn.Module):
    """
    One layer simple linear model
    """

    def __init__(self, D=2, clamp=False):
        self.input_dim = D
        super(LinearModel, self).__init__()
        self.w = nn.Linear(D, 1, bias=True)
        self.w.weight.data.uniform_(-0.0001, 0.0001)
        self.clamp = clamp

    def forward(self, x):
        h = self.w(x)
        return h if not self.clamp else torch.clamp(h, -10, 10)


class CustomLinearModel(nn.Module):
    """
    One layer simple linear model
    with customizability to fix one feature's weight forever,
    use bias term or not etc. Used in the synthetic example of the paper.
    """

    def __init__(self, D=2, clamp=False, use_bias=False, fix_weight_dim=None):
        # doesnt work, write the whole thingy again. should be easy. just do it.
        self.input_dim = D
        super(CustomLinearModel, self).__init__()
        self.fix_weight_dim = fix_weight_dim
        self.w = nn.Linear(
            D - 1, 1,
            bias=use_bias) if fix_weight_dim is not None else nn.Linear(
                D, 1, bias=use_bias)
        self.clamp = clamp

    def forward(self, x):
        dims = list(range(self.input_dim))
        if self.fix_weight_dim is not None:
            dims.remove(self.fix_weight_dim)
        # print(dims)
        x_rest = x[:, dims]
        # print(x_rest)
        h = self.w(x_rest)
        return h if not self.clamp else torch.clamp(h, -10, 10)


class NNModel(nn.Module):
    """
    Neural network model
    """

    def __init__(self,
                 hidden_layer=64,
                 D=2,
                 dropout=0.0,
                 init_weight1=None,
                 init_weight2=None,
                 pooling=False,
                 clamp=False):
        self.input_dim = D
        super(NNModel, self).__init__()
        self.fc = nn.Linear(D, hidden_layer, bias=True)
        self.fc_drop = nn.Dropout(p=dropout)
        # self.activation = nn.ReLU()
        self.activation = nn.ReLU()
        if pooling == "concat_avg":
            self.fc2 = nn.Linear(2 * hidden_layer, hidden_layer, bias=True)
            self.fc3 = nn.Linear(hidden_layer, 1, bias=True)
        elif pooling is not False:
            self.fc2 = nn.Linear(hidden_layer, hidden_layer, bias=True)
            self.fc3 = nn.Linear(hidden_layer, 1, bias=True)
        else:
            self.fc2 = nn.Linear(hidden_layer, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
        if init_weight1 is not None:
            self.fc.weight = torch.nn.Parameter(init_weight1)
        if init_weight2 is not None:
            self.fc2.weight = torch.nn.Parameter(init_weight2)
        self.pooling_layer = pooling
        self.clamp = clamp

    def forward(self, x):
        h = self.activation(self.fc(x))
        h = self.fc_drop(h)
        if self.pooling_layer:
            if self.pooling_layer == "average":
                h1 = h - torch.mean(h, dim=0, keepdim=True)
            elif self.pooling_layer == "max":
                h1 = h - torch.max(h, dim=0, keepdim=True)
            elif self.pooling_layer == "concat_avg":
                h1 = torch.cat(
                    (h, torch.mean(h, dim=0, keepdim=True).repeat(
                        x.size()[0], 1)),
                    dim=1)
        else:
            h1 = h
        h2 = self.fc2(h1)
        if self.pooling_layer:
            h3 = self.fc3(self.activation(h2))
            return h3 if not self.clamp else torch.clamp(h3, -10, 10)

        else:
            return h2 if not self.clamp else torch.clamp(h2, -10, 10)


def convert_to_gpu(model, gpu_id):
    device = torch.device("cuda:" + str(gpu_id))
    return model.to(device)


def convert_vars_to_gpu(varlist, gpu_id):
    device = torch.device("cuda:" + str(gpu_id))
    return [var.to(device) for var in varlist]


if __name__ == "__main__":
    print("Models.py")
