import torch
import torch.nn as nn

class GroupAdditiveCoupling(torch.nn.Module):
    def __init__(self, Fms, group=2):
        super(GroupAdditiveCoupling, self).__init__()

        self.Fms = Fms
        self.group = group

    def forward(self, x, edge_index, *args):
        xs = torch.chunk(x, self.group, dim=-1)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
        args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            y = xs[i] + Fmd
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=-1)

        return out

    def inverse(self, y, edge_index, *args):
        ys = torch.chunk(y, self.group, dim=-1)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
        args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = ys[i-1]
            else:
                y_in = sum(xs)

            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            x = ys[i] - Fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=-1)

        return x
