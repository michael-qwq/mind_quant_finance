import mindspore.ops as ops
import mindspore.nn as nn


class FirstOrderGrad(nn.Cell):

    def __init__(self, model, argnums=0):
        r"""
        Compute First-Order Derivative.

        Args:
            - **model** (nn.Cell) - A function or network that takes Tensor inputs.
            - **argnum** (int) - Specifies which input the output takes the first derivative of. Default: 0.

        Input:
            - **(*x)** (*args) - The input is variable-length argument.

        Output:
            - **grad1** (Tensor) - The first-order grad of the input.
        """
        super(FirstOrderGrad, self).__init__()
        self.model = model
        self.argnums = argnums
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, *x):
        """
        """
        gradient_function = self.grad(self.model)
        gradient = gradient_function(*x)
        output = gradient[self.argnums]
        return output


class SecondOrderGrad(nn.Cell):

    def __init__(self, model, argnums=0):
        r"""
        Compute Second-Order Derivative.

        Args:
            - **model** (nn.Cell) - A function or network that takes Tensor inputs.
            - **argnum** (int) - Specifies which input the output takes the first derivative of. Default: 0.

        Input:
            - **(*x)** (*args) - The input is variable-length argument.

        Output:
            - **grad2** (Tensor) - The second-order grad of the input.
        """
        super(SecondOrderGrad, self).__init__()
        self.grad1 = FirstOrderGrad(model, argnums=argnums)
        self.grad2 = FirstOrderGrad(self.grad1, argnums=argnums)

    def construct(self, *x):
        output = self.grad2(*x)
        return output
