# https://github.com/francois-rozet/piqa/blob/master/piqa/utils.py abaf398
r"""Miscellaneous tools such as modules, functionals and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Union


jit = torch.jit.script


def channel_conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
    padding: int = 0,  # Union[int, Tuple[int, ...]]
) -> torch.Tensor:
    r"""Returns the channel-wise convolution of `x` with respect to `kernel`.
    Args:
        x: An input tensor, (N, C, *).
        kernel: A kernel, (C', 1, *).
        padding: The implicit paddings on both sides of the input dimensions.
    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernel = torch.ones((1, 1, 3, 3))
        >>> channel_conv(x, kernel)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    return F.conv1d(x, kernel, padding=padding, groups=x.size(1))


def channel_sep_conv(
    x: torch.Tensor,
    kernels: List[torch.Tensor],
) -> torch.Tensor:
    r"""Returns the channel-wise convolution of `x` with respect to the
    separated kernel `kernels`.
    Args:
        x: An input tensor, (N, C, *).
        kernels: A separated kernel, (C', 1, 1*, K, 1*).
    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernels = [torch.ones((1, 1, 3, 1)), torch.ones((1, 1, 1, 3))]
        >>> channel_sep_conv(x, kernels)
        tensor([[[[ 54.,  63.,  72.],
                  [ 99., 108., 117.],
                  [144., 153., 162.]]]])
    """

    for kernel in kernels:
        x = channel_conv(x, kernel)

    return x


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
    sigma: float = 1.
) -> torch.Tensor:
    r"""Returns the 1D Gaussian kernel of size `kernel_size`.
    The distribution is centered around the kernel's center
    and the standard deviation is `sigma`.
    Args:
        kernel_size: The size of the kernel.
        sigma: The standard deviation of the distribution.
    Wikipedia:
        https://en.wikipedia.org/wiki/Normal_distribution
    Example:
        >>> gaussian_kernel(5, sigma=1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """

    kernel = torch.arange(kernel_size).float()
    kernel -= (kernel_size - 1) / 2
    kernel = kernel ** 2 / (2. * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel


def haar_kernel(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the separated Haar kernel.
    Args:
        size: The kernel (even) size.
    Wikipedia:
        https://en.wikipedia.org/wiki/Haar_wavelet
    Example:
        >>> haar_kernel(2)
        (tensor([0.5000, 0.5000]), tensor([ 1., -1.]))
    """

    return (
        torch.ones(size) / size,
        torch.tensor([1., -1.]).repeat_interleave(size // 2)
    )


def prewitt_kernel() -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the separated 3x3 Prewitt kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Prewitt_operator
    Example:
        >>> prewitt_kernel()
        (tensor([0.3333, 0.3333, 0.3333]), tensor([ 1.,  0., -1.]))
    """

    return torch.tensor([1., 1., 1.]) / 3, torch.tensor([1., 0., -1.])


def sobel_kernel() -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the separated 3x3 Sobel kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Sobel_operator
    Example:
        >>> sobel_kernel()
        (tensor([0.2500, 0.5000, 0.2500]), tensor([ 1.,  0., -1.]))
    """

    return torch.tensor([1., 2., 1.]) / 4, torch.tensor([1., 0., -1.])


def scharr_kernel() -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the separated 3x3 Scharr kernel.
    Wikipedia:
        https://en.wikipedia.org/wiki/Scharr_operator
    Example:
        >>> scharr_kernel()
        (tensor([0.1875, 0.6250, 0.1875]), tensor([ 1.,  0., -1.]))
    """

    return torch.tensor([3., 10., 3.]) / 16, torch.tensor([1., 0., -1.])


def tensor_norm(
    x: torch.Tensor,
    dim: List[int],  # Union[int, Tuple[int, ...]] = ()
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
    dim: List[int],  # Union[int, Tuple[int, ...]] = ()
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
        self.targets = targets
        self.len = len(self.targets)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        r"""Defines the computation performed at every call.
        """

        output = []
        j = 0

        for i, layer in enumerate(self.layers):
            input = layer(input)

            if i == self.targets[j]:
                output.append(input)
                j += 1

                if j == self.len:
                    break

        return output


def build_reduce(reduction: str = 'mean') -> nn.Module:
    r"""Returns a reducing module.
    Args:
        reduction: Specifies the reduce type:
            `'none'` | `'mean'` | `'sum'`.
    Example:
        >>> red = build_reduce(reduction='sum')
        >>> red(torch.arange(5))
        tensor(10)
    """

    if reduction == 'mean':
        return _Mean()
    elif reduction == 'sum':
        return _Sum()

    return nn.Identity()


class _Mean(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.mean()


class _Sum(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.sum()


# https://github.com/francois-rozet/piqa/blob/master/piqa/ssim.py abaf398

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

# from piqa.utils import jit, build_reduce, gaussian_kernel, channel_sep_conv

from typing import Union, List, Tuple

_MS_WEIGHTS = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


@jit
def create_window(
        window_size: int,
        n_channels: int,
        device: torch.device = torch.device('cpu'),
) -> List[torch.Tensor]:
    r"""Returns the SSIM convolution window of size `window_size`.
    Args:
        window_size: The size of the window.
        n_channels: A number of channels.
        device: Specifies the device of the created window.
    Example:
        >>> win = create_window(5, n_channels=3)
        >>> win[0].size()
        torch.Size([3, 1, 5, 1])
        >>> win[0][0]
        tensor([[[0.1201],
                 [0.2339],
                 [0.2921],
                 [0.2339],
                 [0.1201]]])
    """

    kernel = gaussian_kernel(window_size, 1.5).to(device)
    kernel = kernel.repeat(n_channels, 1, 1)

    return [
        kernel.unsqueeze(-1).contiguous(),
        kernel.unsqueeze(-2).contiguous()
    ]


@jit
def ssim_per_channel(
        x: torch.Tensor,
        y: torch.Tensor,
        window: List[torch.Tensor],
        value_range: float = 1.,
        non_negative: bool = False,
        k1: float = 0.01,
        k2: float = 0.03,
) -> List[torch.Tensor]:
    r"""Returns the SSIM and the contrast sensitivity per channel
    between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A separated kernel, ((C, 1, K, 1), (C, 1, 1, K)).
        value_range: The value range of the inputs (usually 1. or 255).
        non_negative: Whether negative values are clipped or not.
        For the remaining arguments, refer to [2].
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
    mu_x = channel_sep_conv(x, window)
    mu_y = channel_sep_conv(y, window)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Variance (sigma)
    sigma_xx = channel_sep_conv(x ** 2, window) - mu_xx
    sigma_yy = channel_sep_conv(y ** 2, window) - mu_yy
    sigma_xy = channel_sep_conv(x * y, window) - mu_xy

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

    window = create_window(window_size, x.size(1), device=x.device)

    return ssim_per_channel(x, y, window, **kwargs)[0].mean(-1)


@jit
def msssim_per_channel(
        x: torch.Tensor,
        y: torch.Tensor,
        window: List[torch.Tensor],
        weights: torch.Tensor,
        value_range: float = 1.,
        k1: float = 0.01,
        k2: float = 0.03,
) -> torch.Tensor:
    """Returns the MS-SSIM per channel between `x` and `y`.
    Args:
        x: An input tensor, (N, C, H, W).
        y: A target tensor, (N, C, H, W).
        window: A separated kernel, ((C, 1, K, 1), (C, 1, 1, K)).
        weights: The weights of the scales, (M,).
        value_range: The value range of the inputs (usually 1. or 255).
        For the remaining arguments, refer to [2].
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

    m = weights.numel()
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)

        ss, cs = ssim_per_channel(
            x, y, window,
            value_range=value_range,
            non_negative=True,
            k1=k1, k2=k2,
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1)
    msss = (msss ** weights).prod(dim=-1)

    return msss


def msssim(
        x: torch.Tensor,
        y: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
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

    window = create_window(window_size, x.size(1), device=x.device)

    if weights is None:
        weights = _MS_WEIGHTS.to(x.device)

    return msssim_per_channel(x, y, window, weights, **kwargs).mean(-1)


class SSIM(nn.Module):
    r"""Creates a criterion that measures the SSIM
    between an input and a target.
    Args:
        window_size: The size of the window.
        n_channels: The number of channels.
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

        window = create_window(window_size, n_channels)

        self.register_buffer('window0', window[0])
        self.register_buffer('window1', window[1])

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    @property
    def window(self) -> List[torch.Tensor]:
        return [self.window0, self.window1]

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
        n_channels: The number of channels.
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

        window = create_window(window_size, n_channels)

        self.register_buffer('window0', window[0])
        self.register_buffer('window1', window[1])

        if weights is None:
            weights = _MS_WEIGHTS

        self.register_buffer('weights', weights)

        self.reduce = build_reduce(reduction)
        self.kwargs = kwargs

    @property
    def window(self) -> List[torch.Tensor]:
        return [self.window0, self.window1]

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