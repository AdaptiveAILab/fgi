# Flexible and Efficient Surrogate Gradient Modeling with Forward Gradient Injection
Sebastian Otte<br>
[Adaptive AI Lab](https://www.adaptiveailab.com), University of Lübeck

## Abstract

Automatic differentiation is a key feature of present deep learning frameworks. Moreover, they typically provide various ways to specify custom gradients within the computation graph, which is of particular importance for defining surrogate gradients in the realms of non-differentiable operations such as the Heaviside function in spiking neural networks (SNNs). PyTorch, for example, allows the custom specification of the backward pass of an operation by overriding its backward method. Other frameworks provide comparable options. While these methods are common practice and usually work well, they also have several disadvantages such as limited flexibility, additional source code overhead, poor usability, or a potentially strong negative impact on the effectiveness of automatic model optimization procedures. Here, an alternative way to formulate surrogate gradients is presented, namely, forward gradient injection (FGI). FGI applies a simple but effective combination of basic standard operations to inject an arbitrary gradient shape into the computational graph directly within the forward pass. 

## Standard way of modeling surrogate gradient functions
The standard way of modeling a surrogate gradient function (e.g. in PyTorch) is overriding the backward method of a module or an autograd function as shown below.

```python
class StepGaussianGrad(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = gaussian(x)
        return grad_output * dfdx
```

This way we can equipt a non differentiable function (e.g. the Heaviside function) with a derivate such that back prop can be applied.

```python
x = torch.linspace(-3, 3, 1000, requires_grad = True)

# Compute outputs
y = StepGaussianGrad.apply(x)

# Backprop
y.sum().backward()
dydx = x.grad
```

![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/25141a16-1586-40fc-bf9f-8e41ac2d67f2)


### While this undergoing typically fulfills what is needed, it has several drawbacks:

- It comes with the price of a significant amount of code overhead
- If a custom gradient implementation is required on-the-fly this might disrupt the workflow, affect code readability and compatibility, and complicate model prototyping
- Moreover, this may also block framework specific builtin optimization routines, such as TorchScript

## Forward gradient injection (FGI)

With forward gradient injection (FGI) we can model surrogate gradient functions directly inline within the forward pass.

Let $x$ be a tensor of interest, $f(x)$ an operation for which we want to substitute the gradient, and let $g'(x)$ be the shape of the desired surrogate derivative. We formulate FGI through:
$$h = x \cdot \text{sg}(g'(x))$$
$$y = h − \text{sg}(h) + \text{sg}(f(x))$$
where $\text{sg}$ is the stop gradient operator. 

- The forward pass will produce $y = f(x)$ due to out-canceling.

- When we now compute the derivative of $y$ with respect to $x$ in the backward pass we obtain: $$\frac{\partial y}{\partial x} = g'(x)$$

For details see the [paper](https://arxiv.org/pdf/2406.00177).

<br>

FGI in PyTorch can be realized as follows (here we to use a double Gaussian function as surrogate gradient):

```python
# Generate inputs
x = torch.linspace(-3, 3, 1000, requires_grad = True)

# Apply FGI and compute outputs:
mul = x * dblgaussian(x).detach()
y = mul - mul.detach() + step(x).detach()

# Backprop
y.sum().backward()
dydx = x.grad
```

![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/b9e7cfd2-8939-4d86-a5b2-596dc51b6e8c)


## FGI provides performance advantages for automatic model optimization

Using FGI instead of overriding the backward method can provide significant advantages for automatic model optimization routines (here shown for TorchScript and torch.comile).

![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/0db6e3c8-8ac2-4872-ac75-2502d4760e95)
![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/ecf3f4a9-2106-416e-bceb-d8f6ee63efb3)
![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/19d3d26d-3db5-493b-9697-afa20972a0ce)
![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/faad2932-7881-4afb-947e-e269a0e02c9d)
![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/10410559-7ddd-4432-8b5a-79b91886c961)

Note that with increasing sequence length, torch.compile has extended warmup costs. More details and results can be found in the [paper](https://arxiv.org/pdf/2406.00177).

## See FGI in action

Applying FGI in the context of recent [balanced resonate-and-fire (BRF) neurons](https://openreview.net/forum?id=dkdilv4XD4) within recurrent spiking networks results in significant speedups for TorchScript:

BRF-RSNN training speed up with model optimization methods of FGI over standard backward() baseline.
![image](https://github.com/AdaptiveAILab/fgi/assets/3898842/bef06d00-19f3-480d-8317-511cccc2a961)

See this [paper](https://openreview.net/forum?id=dkdilv4XD4) for details.



## Publication and BibTeX

If you find this repository helpful and use FGI for your research, please cite:

- Sebastian Otte (2024). **Flexible and Efficient Surrogate Gradient Modeling with Forward Gradient Injection**. *First Austrian Symposium on AI, Robotics, and Vision*. innsbruck university press. arXiv preprint [arXiv:2406.00177](https://arxiv.org/abs/2406.00177).

```
@InProceedings{Otte2024Flexible,
    author        = {Sebastian Otte},
    title         = {Flexible and Efficient Surrogate Gradient Modeling with Forward Gradient Injection},
    year          = {2024},
    booktitle     = {First Austrian Symposium on AI, Robotics, and Vision},
    publisher     = {innsbruck university press},
    doi           = {10.15203/99106-150-2-74},
    eprint        = {2406.00177},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```
