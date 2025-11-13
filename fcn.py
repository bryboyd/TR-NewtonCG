import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, in_neurons, hid_neurons, n_hidden_layers, out_neurons):
        """
        Create a fully-connected network (i.e., multi-layer perceptron) with specified number of neurons and layers.

        Parameters
        ----------
        in_neurons      - Number of input neurons
        hid_neurons     - Number of neurons per hidden layer
        n_hidden_layers - Number of hidden layers
        out_neurons     - Number of output neurons
        """
        super().__init__()
        activation = nn.Tanh

        self.input_layer = nn.Sequential(nn.Linear(in_neurons, hid_neurons), activation())
        self.hidden_layers = nn.Sequential(*[self._generate_hidden(hid_neurons, activation)
                                             for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(hid_neurons, out_neurons)


    def forward(self, x):
        a = self.input_layer(x)
        z = self.hidden_layers(a)
        y = self.output_layer(z)

        return y


    def _generate_hidden(self, hid_neurons, activation):
        return nn.Sequential(nn.Linear(hid_neurons, hid_neurons), activation())


if __name__ == "__main__":
    torch.manual_seed(42)
    t = torch.rand((3))
    model = FCN(3, 3, 2, 1)
    model(t)

    # for name, param in model.named_parameters():
    #     print(f"name: {name}, param shape: {param.shape}")

    # print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Total weights: ", sum(p.numel() for name, p in model.named_parameters() if "weight" in name))