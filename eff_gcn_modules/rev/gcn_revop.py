"""This module is implemented by Guohao Li based on MemCNN @ Copyright (c) 2018 Sil C. van de Leemput under MIT license."""

import numpy as np
import torch
import torch.nn as nn

use_context_mans = True

try:
    pytorch_version_one_and_above = int(torch.__version__[0]) > 0
except TypeError:
    pytorch_version_one_and_above = True


class InvertibleCheckpointFunction(torch.autograd.Function):
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
            raise RuntimeError("InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpointFunction for more than once.")
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        with torch.random.fork_rng(devices=rng_devices, enabled=False):
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
    def __init__(self, fn):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.

        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`
        """
        super(InvertibleModuleWrapper, self).__init__()
        self._fn = fn

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
        y = InvertibleCheckpointFunction.apply(
            self._fn.forward,
            self._fn.inverse,
            len(xin),
            *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])))

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`

        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).

        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.

        """
        x = InvertibleCheckpointFunction.apply(
            self._fn.inverse,
            self._fn.forward,
            len(yin),
            *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])))

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x

# To consider:  maybe get_device_states and set_device_states should reside in
# torch/random.py?
#
# get_device_states and set_device_states cannot be imported from
# torch.utils.checkpoint, since it was not
# present in older versions, so we include a copy here.
def get_device_states(*args):
      # This will not error out if "arg" is a CPU tensor or a non-tensor type
      # because
      # the conditionals short-circuit.
      fwd_gpu_devices = list(set(arg.get_device() for arg in args
                            if isinstance(arg, torch.Tensor) and arg.is_cuda))

      fwd_gpu_states = []
      for device in fwd_gpu_devices:
          with torch.cuda.device(device):
              fwd_gpu_states.append(torch.cuda.get_rng_state())

      return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states):
      for device, state in zip(devices, states):
          with torch.cuda.device(device):
              torch.cuda.set_rng_state(state)
