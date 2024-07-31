from torch import nn


class MLP(nn.Module):
    def __init__(self, widths, act_fun=nn.ReLU, activate_final=None):
        super().__init__()

        layers = []

        n_in = widths[0]
        for i, w in enumerate(widths[1:-1]):
            linear = nn.Linear(n_in, w)
            layers.append(linear)

            act = act_fun()
            layers.append(act)

            n_in = w

        linear = nn.Linear(n_in, widths[-1])
        layers.append(linear)

        if activate_final is not None:
            act = activate_final()
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
