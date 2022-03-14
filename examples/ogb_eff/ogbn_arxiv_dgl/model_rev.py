import __init__
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from eff_gcn_modules.rev.rev_layer import SharedDropout
from copy import deepcopy

class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0 # not implemented
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0 # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class RevGATBlock(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        edge_emb,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(RevGATBlock, self).__init__()

        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = GATConv(
                        node_feats,
                        out_feats,
                        num_heads=n_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        negative_slope=negative_slope,
                        residual=residual,
                        activation=activation,
                        use_attn_dst=use_attn_dst,
                        allow_zero_in_degree=allow_zero_in_degree,
                        use_symmetric_norm=use_symmetric_norm,
                    )
        self.dropout = SharedDropout()
        if edge_emb > 0:
            self.edge_encoder = nn.Linear(edge_feats, edge_emb)
        else:
            self.edge_encoder = None

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        if isinstance(self.dropout, SharedDropout):
            self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)

        if self.edge_encoder is not None:
            if efeat is None:
                efeat = graph.edata["feat"]
            efeat_emb = self.edge_encoder(efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)
        else:
            efeat_emb = None

        out = self.conv(graph, out, perm).flatten(1, -1)
        return out

class GroupAdditiveCoupling(torch.nn.Module):
    def __init__(self, fm, group=2):
        super(GroupAdditiveCoupling, self).__init__()

        self.Fms = nn.ModuleList()
        for i in range(group):
            if i == 0:
                self.Fms.append(fm)
            else:
                self.Fms.append(deepcopy(fm))

        self.group = group

    def forward(self, x, edge_index, *args):
        xs = torch.chunk(x, self.group, dim=-1)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
        args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i](y_in, edge_index, *args_chunks[i])
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

            Fmd = self.Fms[i](y_in, edge_index, *args_chunks[i])
            x = ys[i] - Fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=-1)

        return x

class InvertibleCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]

        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                else:
                    x.append(element)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        # only clear memory of node features
        inputs[0].storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs]
        ctx.outputs = [detached_outputs]

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpoint is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpoint for more than once.")
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input
        with torch.no_grad():
            # edge_index and edge_emb
            inputs_inverted = ctx.fn_inverse(*(outputs+inputs[1:]))
            # clear memory from outputs
            for element in outputs:
                element.storage().resize_(0)

            if not isinstance(inputs_inverted, tuple):
                inputs_inverted = (inputs_inverted,)
            for element_original, element_inverted in zip(inputs, inputs_inverted):
                element_original.storage().resize_(int(np.prod(element_original.size())))
                element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for det_input, requires_grad in zip(detached_inputs, ctx.input_requires_grad):
                det_input.requires_grad = requires_grad
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        filtered_detached_inputs = tuple(filter(lambda x: x.requires_grad,
                                               detached_inputs))
        gradients = torch.autograd.grad(outputs=temp_output,
                                        inputs=filtered_detached_inputs + ctx.weights,
                                        grad_outputs=grad_outputs)

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(self, fm, group=2):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.
        """
        super(InvertibleModuleWrapper, self).__init__()
        self.Fms = nn.ModuleList()
        for i in range(group):
            if i == 0:
                self.Fms.append(fm)
            else:
                self.Fms.append(deepcopy(fm))
        self.group = group

    def _forward(self, x, edge_index, *args):
        xs = torch.chunk(x, self.group, dim=-1)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
        args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i](y_in, edge_index, *args_chunks[i])
            y = xs[i] + Fmd
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=-1)

        return out

    def _inverse(self, y, edge_index, *args):
        ys = torch.chunk(y, self.group, dim=-1)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
        args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = ys[i-1]
            else:
                y_in = sum(xs)

            Fmd = self.Fms[i](y_in, edge_index, *args_chunks[i])
            x = ys[i] - Fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=-1)

        return x

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`

        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.

        """
        y = InvertibleCheckpoint.apply(
            self._forward,
            self._inverse,
            len(xin),
            *(xin + tuple([p for p in self.parameters() if p.requires_grad])))

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

class RevGAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        group=2,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.group = group

        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(n_heads * n_hidden)

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1

            if i == 0 or i == n_layers -1:
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                    )
                )
            else:
                fm = RevGATBlock(
                    in_hidden // group,
                    0,
                    0,
                    out_hidden // group,
                    n_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
                # invertible_module = GroupAdditiveCoupling(fm, group=self.group)

                # conv = InvertibleModuleWrapper(fn=invertible_module)
                conv = InvertibleModuleWrapper(fm, group=self.group)
                self.convs.append(conv)

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(),
                                  device=graph.device)
            self.perms.append(perm)

        h = self.convs[0](graph, h, self.perms[0]).flatten(1, -1)

        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for i in range(1, self.n_layers-1):
            graph.requires_grad = False
            perm = torch.stack([self.perms[i]]*self.group, dim=1)
            h = self.convs[i](h, graph, mask, perm)

        h = self.norm(h)
        h = self.activation(h, inplace=True)
        h = self.dp_last(h)
        h = self.convs[-1](graph, h, self.perms[-1])

        h = h.mean(1)
        h = self.bias_last(h)

        return h
