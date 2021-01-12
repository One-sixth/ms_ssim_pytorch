# ms_ssim_pytorch

The code was modified from https://github.com/VainF/pytorch-msssim.  
Part of the code has been modified to make it faster, takes up less VRAM, and is compatible with pytorch jit.  

The dynamic channel version can found here https://github.com/One-sixth/ms_ssim_pytorch/tree/dynamic_channel_num.  
More convenient to use but has a little performance loss.  

Thanks [vegetable09](https://github.com/vegetable09) for finding and fixing a bug that causes gradient nan when ms_ssim backward. [#3](https://github.com/One-sixth/ms_ssim_pytorch/issues/3)  

If you are using pytorch 1.2, please be careful not to create and destroy this jit module in the training loop (other jit modules may also have this situation), there may be memory leaks. I have tested that pytorch 1.6 does not have this problem. [#4](https://github.com/One-sixth/ms_ssim_pytorch/issues/4)

# Speed up. Only test on GPU.
losser1 is https://github.com/lizhengwei1992/MS_SSIM_pytorch/blob/master/loss.py 268fc76  
losser2 is https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py 881d210  
losser3 is https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py b47c07c  
losser4 is https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim.py 0f69f16  

In pytorch 1.1 1.2  
My test environment: i7-6700HQ GTX970M-3G  

## SSIM
Test output  

pytorch 1.7.1  
```
Performance Testing SSIM

testing losser2
cuda time 40721.16796875
perf_counter time 36.6222991

testing losser3
cuda time 17215.404296875
perf_counter time 17.1855524

testing losser4
cuda time 14191.6328125
perf_counter time 11.753846000000003

testing losser5
cuda time 39380.390625
perf_counter time 35.5724254

```

pytorch 1.2  
```
Performance Testing SSIM

testing losser2
cuda time 89290.7734375
perf_counter time 87.1042247

testing losser3
cuda time 36153.64453125
perf_counter time 36.09167939999999

testing losser4
cuda time 31085.455078125
perf_counter time 29.80807200000001

```

pytorch 1.1  
```
Performance Testing SSIM

testing losser2
cuda time 88990.0703125
perf_counter time 86.80163019999999

testing losser3
cuda time 36119.06640625
perf_counter time 36.057978399999996

testing losser4
cuda time 34708.8359375
perf_counter time 33.916086199999995

```

## MS-SSIM
Test output  

pytorch 1.7.1  
```
Performance Testing MS_SSIM

testing losser1
cuda time 58361.2265625
perf_counter time 58.3090031

testing losser3
cuda time 26812.125
perf_counter time 26.7919251

testing losser4
cuda time 25492.28125
perf_counter time 25.485101200000003

testing losser5
cuda time 52880.6015625
perf_counter time 52.83433840000001

```

pytorch 1.2  
```
Performance Testing MS_SSIM

testing losser1
cuda time 134158.84375
perf_counter time 134.0433756

testing losser3
cuda time 62143.4140625
perf_counter time 62.103911400000015

testing losser4
cuda time 46854.25390625
perf_counter time 46.81785239999999

```

pytorch 1.1  
```
Performance Testing MS_SSIM

testing losser1
cuda time 134115.96875
perf_counter time 134.0006031

testing losser3
cuda time 61760.56640625
perf_counter time 61.71994470000001

testing losser4
cuda time 52888.03125
perf_counter time 52.848280500000016

```

## Test speed by yourself
1. cd ms_ssim_pytorch/_test_speed  

2. python test_ssim_speed.py  
or  
2. python test_ms_ssim_speed.py  

# Other thing
Add parameter use_padding.  
When set to True, the gaussian_filter behavior is the same as https://github.com/Po-Hsun-Su/pytorch-ssim.  
This parameter is mainly used for MS-SSIM, because MS-SSIM needs to be downsampled.  
When the input image is smaller than 176x176, this parameter needs to be set to True to ensure that MS-SSIM works normally. (when parameter weight and level are the default)  

# Require
Pytorch >= 1.1  

if you want to test the code with animation. You also need to install some package.  
```
pip install imageio imageio-ffmpeg opencv-python
```

# Test code with animation
The test code is included in the ssim.py file, you can run the file directly to start the test.  

1. git clone https://github.com/One-sixth/ms_ssim_pytorch  
2. cd ms_ssim_pytorch  
3. python ssim.py  

# Code Example.
```python
import torch
import ssim


im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
img1 = im / 255
img2 = img1 * 0.5

losser = ssim.SSIM(data_range=1., channel=3).cuda()
loss = losser(img1, img2).mean()

losser2 = ssim.MS_SSIM(data_range=1., channel=3).cuda()
loss2 = losser2(img1, img2).mean()

print(loss.item())
print(loss2.item())
```

# Animation
GIF is a bit big. Loading may take some time.  
Or you can download the mkv video file directly to view it, smaller and smoother.  
https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim_test.mkv  
https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ms_ssim_test.mkv  

SSIM  
![ssim](https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim_test.gif)

MS-SSIM  
![ms-ssim](https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ms_ssim_test.gif)

# References
https://github.com/VainF/pytorch-msssim  
https://github.com/Po-Hsun-Su/pytorch-ssim  
https://github.com/lizhengwei1992/MS_SSIM_pytorch  
