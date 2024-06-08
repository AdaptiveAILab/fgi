import sys
import torch


class SimpleLSTM(torch.nn.Module):

    def __init__(
            self, input_size: int, hidden_size: int, output_size: int
    ):
        super(SimpleLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        hx = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        cx = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        state = (hx, cx)

        for t in range(sequence_length):
            input_t = x[t]

            hx, cx = self.lstm(input_t, state)
            state = (hx, cx)

            out = self.output_layer(hx)
            outputs.append(out)

        return torch.stack(outputs)


class SNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            hidden_bias: bool = False,
            output_bias: bool = False
    ) -> None:
        super(SNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden = ALIFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            adaptive_alpha=True,
            adaptive_rho=True,
            bias=hidden_bias
        )

        self.out = LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_alpha=True,
            bias=output_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):
            input_t = x[t]
            hidden = hidden_z, hidden_u, hidden_a

            hidden_z, hidden_u, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            out_u = self.out(hidden_z, out_u)
            outputs.append(out_u)

        return torch.stack(outputs)

