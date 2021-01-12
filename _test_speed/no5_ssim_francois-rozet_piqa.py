# https://github.com/francois-rozet/piqa/blob/master/piqa/utils.py 0b5a85e
r"""Miscellaneous tools such as modules, functionals and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Union


def build_reduce(
    reduction: str = 'mean',
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns a reduce function.
    Args:
        reduction: Specifies the reduce function:
            `'none'` | `'mean'` | `'sum'`.
    Example:
        >>> red = build_reduce(reduction='sum')
        >>> callable(red)
        True
        >>> red(torch.arange(5))
        tensor(10)
    """

    if reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum

    return nn.Identity()


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).
    Returns:
        coord: The unraveled coordinates, (*, N, D).
    Example:
        >>> unravel_index(torch.arange(9), (3, 3))
        tensor([[0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2]])
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def gaussian_kernel(
    kernel_size: int,
    sigma: float = 1.,
    n: int = 2,
) -> torch.Tensor:
    r"""Returns the `n`-dimensional Gaussian kernel of size `kernel_size`.
    The distribution is centered around the kernel's center
    and the standard deviation is `sigma`.
    Args:
        kernel_size: The size of the kernel.
        sigma: The standard deviation of the distribution.
        n: The number of dimensions of the kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Normal_distribution
    Example:
        >>> gaussian_kernel(5, sigma=1.5, n=2)
        tensor([[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]])
    """

    shape = (kernel_size,) * n

    kernel = unravel_index(
        torch.arange(kernel_size ** n),
        shape,
    ).float()

    kernel -= (kernel_size - 1) / 2
    kernel = (kernel ** 2).sum(1) / (2. * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel.reshape(shape)


def filter2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    padding: Union[int, Tuple[int, int]] = 0,
) -> torch.Tensor:
    r"""Returns the 2D (channel-wise) filter of `x` with respect to `kernel`.
    Args:
        x: An input tensor, (N, C, H, W).
        kernel: A 2D filter kernel, (C', 1, K, L).
        padding: The implicit paddings on both sides of the input.
    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernel = gaussian_kernel(3, sigma=1.5)
        >>> kernel
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> filter2d(x, kernel.view(1, 1, 3, 3))
        tensor([[[[ 6.0000,  7.0000,  8.0000],
                  [11.0000, 12.0000, 13.0000],
                  [16.0000, 17.0000, 18.0000]]]])
    """

    return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


def haar_kernel(size: int):
    r"""Returns the (horizontal) Haar kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Haar_wavelet
    Example:
        >>> haar_kernel(2)
        tensor([[ 0.5000, -0.5000],
                [ 0.5000, -0.5000]])
    """

    kernel = torch.ones((size, size)) / size
    kernel[:, size // 2:] *= -1

    return kernel


def prewitt_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Prewitt kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Prewitt_operator
    Example:
        >>> prewitt_kernel()
        tensor([[ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333]])
    """

    return torch.Tensor([
        [1., 0., -1.],
        [1., 0., -1.],
        [1., 0., -1.],
    ]) / 3


def sobel_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Sobel kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Sobel_operator
    Example:
        >>> sobel_kernel()
        tensor([[ 0.2500,  0.0000, -0.2500],
                [ 0.5000,  0.0000, -0.5000],
                [ 0.2500,  0.0000, -0.2500]])
    """

    return torch.Tensor([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.],
    ]) / 4


def scharr_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Scharr kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Scharr_operator
    Example:
        >>> scharr_kernel()
        tensor([[ 0.1875,  0.0000, -0.1875],
                [ 0.6250,  0.0000, -0.6250],
                [ 0.1875,  0.0000, -0.1875]])
    """

    return torch.Tensor([
        [3., 0., -3.],
        [10., 0., -10.],
        [3., 0., -3.],
    ]) / 16


def tensor_norm(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = (),
    keepdim: bool = False,
    norm: str = 'L2',
) -> torch.Tensor:
    r"""Returns the norm of `x`.
    Args:
        x: An input tensor.
        dim: The dimension(s) along which to calculate the norm.
        keepdim: Whether the output tensor has `dim` retained or not.
        norm: Specifies the norm funcion to apply:
            `'L1'` | `'L2'` | `'L2_squared'`.
    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)
    Example:
        >>> x = torch.arange(9).float().view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> tensor_norm(x, dim=0)
        tensor([6.7082, 8.1240, 9.6437])
    """

    if norm in ['L2', 'L2_squared']:
        x = x ** 2
    else:  # norm == 'L1'
        x = x.abs()

    x = x.sum(dim=dim, keepdim=keepdim)

    if norm == 'L2':
        x = x.sqrt()

    return x


def normalize_tensor(
    x: torch.Tensor,
    dim: Tuple[int, ...] = (),
    norm: str = 'L2',
    epsilon: float = 1e-8,
) -> torch.Tensor:
    r"""Returns `x` normalized.
    Args:
        x: An input tensor.
        dim: The dimension(s) along which to normalize.
        norm: Specifies the norm funcion to use:
            `'L1'` | `'L2'` | `'L2_squared'`.
        epsilon: A numerical stability term.
    Example:
        >>> x = torch.arange(9).float().view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> normalize_tensor(x, dim=0)
        tensor([[0.0000, 0.1231, 0.2074],
                [0.4472, 0.4924, 0.5185],
                [0.8944, 0.8616, 0.8296]])
    """

    norm = tensor_norm(x, dim=dim, keepdim=True, norm=norm)

    return x / (norm + epsilon)


def cpow(
    x: torch.cfloat,
    exponent: Union[int, float, torch.Tensor],
) -> torch.cfloat:
    r"""Returns the power of `x` with `exponent`.
    Args:
        x: A complex input tensor.
        exponent: The exponent value or tensor.
    Example:
        >>> x = torch.tensor([1. + 0.j, 0.707 + 0.707j])
        >>> cpow(x, 2)
        tensor([ 1.0000e+00+0.0000j, -4.3698e-08+0.9997j])
    """

    r = x.abs() ** exponent
    phi = torch.atan2(x.imag, x.real) * exponent

    return torch.complex(r * torch.cos(phi), r * torch.sin(phi))


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate
    target layers of a sequential module during its forward pass.
    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        r""""""
        super().__init__()

        self.layers = layers
        self.targets = set(targets)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        r"""Defines the computation performed at every call.
        """

        output = []

        for i, layer in enumerate(self.layers):
            input = layer(input)

            if i in self.targets:
                output.append(input.clone())

            if len(output) == len(self.targets):
                break

        return output


# https://github.com/francois-rozet/piqa/blob/master/piqa/ssim.py 539fc35
r"""Structural Similarity (SSIM) and Multi-Scale Structural Similarity (MS-SSIM)
This module implements the SSIM and MS-SSIM in PyTorch.
Wikipedia:
    https://en.wikipedia.org/wiki/Structural_similarity
Credits:
    Inspired by [pytorch-msssim](https://github.com/VainF/pytorch-msssim)
References:
    [1] Multiscale structural similarity for image quality assessment
    (Wang et al., 2003)
    https://ieeexplore.ieee.org/abstract/document/1292216/
    [2] Image quality assessment: From error visibility to structural similarity
    (Wang et al., 2004)
    https://ieeexplore.ieee.org/abstract/document/1284395/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from piqa.utils import build_reduce, gaussian_kernel, filter2d

from typing import Tuple

_MS_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def create_window(window_size: int, n_channels: int) -> torch.Tensor:
    r"""Returns the SSIM convolution window (kernel) of size `window_size`.
    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
    Example:
        >>> win = create_window(5, n_channels=3)
        >>> win.size()
        torch.Size([3, 1, 5, 5])
        >>> win[0]
        tensor([[[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                 [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                 [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]]])
    """

    kernel = gaussian_kernel(window_size, 1.5)
    window = kernel.repeat(n_channels, 1, 1, 1)

    return window


def ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    value_range: float = 1.,
    non_negative: bool = False,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the SSIM and the contrast sensitivity per channel
    between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window, (C, 1, K, K).
        value_range: The value range of the inputs (usually 1. or 255).
        non_negative: Whether negative values are clipped or not.
        For the remaining arguments, refer to [1].
    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = create_window(7, 3)
        >>> ss, cs = ssim_per_channel(x, y, window)
        >>> ss.size(), cs.size()
        (torch.Size([5, 3]), torch.Size([5, 3]))
    """

    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2

    # Mean (mu)
    mu_x = filter2d(x, window)
    mu_y = filter2d(y, window)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = filter2d(x ** 2, window) - mu_xx
    sigma_yy = filter2d(y ** 2, window) - mu_yy
    sigma_xy = filter2d(x * y, window) - mu_xy

    # Contrast sensitivity
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    # Average
    ss, cs = ss.mean((-1, -2)), cs.mean((-1, -2))

    if non_negative:
        ss, cs = torch.relu(ss), torch.relu(cs)

    return ss, cs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the SSIM between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.
        `**kwargs` are transmitted to `ssim_per_channel`.
    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = ssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    return ssim_per_channel(x, y, window, **kwargs)[0].mean(-1)


def msssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    weights: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A convolution window, (C, 1, K, K).
        weights: The weights of the scales, (M,).
        `**kwargs` are transmitted to `ssim_per_channel`, with
        the exception of `non_negative`.
    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> window = create_window(7, 3)
        >>> weights = torch.rand(5)
        >>> l = msssim_per_channel(x, y, window, weights)
        >>> l.size()
        torch.Size([5, 3])
    """

    css = []

    for i in range(weights.numel()):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = ssim_per_channel(x, y, window, non_negative=True, **kwargs)
        css.append(cs)

    msss = torch.stack(css[:-1] + [ss], dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss


def msssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    weights: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    r"""Returns the MS-SSIM between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window_size: The size of the window.
        weights: The weights of the scales, (M,).
            If `None`, use the official weights instead.
        `**kwargs` are transmitted to `msssim_per_channel`.
    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = msssim(x, y)
        >>> l.size()
        torch.Size([5])
    """

    n_channels = x.size(1)
    window = create_window(window_size, n_channels).to(x.device)

    if weights is None:
        weights = _MS_WEIGHTS.to(x.device)

    return msssim_per_channel(x, y, window, weights, **kwargs).mean(-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.
    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.
        `**kwargs` are transmitted to `ssim_per_channel`.
    Shape:
        * Input: (N, C, H, W)
        * Target: (N, C, H, W), same shape as the input
        * Output: (N,) or (1,) depending on `reduction`
    Example:
        >>> criterion = SSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        window_size: int = 11,
        n_channels: int = 3,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        self.register_buffer('window', create_window(window_size, n_channels))

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = ssim_per_channel(
            input,
            target,
            window=self.window,
            **self.kwargs,
        )[0].mean(-1)

        return self.reduce(l)


class MSSSIM(nn.Module):
    r"""Creates a criterion that measures the MS-SSIM
    between an input and a target.
    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        weights: The weights of the scales, (M,).
            If `None`, use the official weights instead.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.
        `**kwargs` are transmitted to `msssim_per_channel`.
    Shape:
        * Input: (N, C, H, W)
        * Target: (N, C, H, W), same shape as the input
        * Output: (N,) or (1,) depending on `reduction`
    Example:
        >>> criterion = MSSSIM().cuda()
        >>> x = torch.rand(5, 3, 256, 256).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
    """

    def __init__(
        self,
        window_size: int = 11,
        n_channels: int = 3,
        weights: torch.Tensor = None,
        reduction: str = 'mean',
        **kwargs,
    ):
        r""""""
        super().__init__()

        if weights is None:
            weights = _MS_WEIGHTS

        self.register_buffer('window', create_window(window_size, n_channels))
        self.register_buffer('weights', weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        l = msssim_per_channel(
            input,
            target,
            window=self.window,
            weights=self.weights,
            **self.kwargs,
        ).mean(-1)

        return self.reduce(l)