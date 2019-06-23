# ms_ssim_pytorch

The code was modified from https://github.com/VainF/pytorch-msssim.  
Part of the code has been modified to make it faster, takes up less VRAM, and is compatible with pytorch jit.  

# Speed up. Only test on GPU.
## SSIM

## MS-SSIM

# Other thing
Add parameter use_padding.  
When set to True, the gaussian_filter behavior is the same as https://github.com/Po-Hsun-Su/pytorch-ssim.  
This parameter is mainly used for MS-SSIM, because MS-SSIM needs to be downsampled.  
When the input image is smaller than 176x176, this parameter needs to be set to True to ensure that MS-SSIM works normally. (when parameter weight and level are the default)  

# Require
Pytorch >= 1.1  

if you want to test the code. You also need to install some package.  
```
pip install imageio imageio-ffmpeg opencv-python
```

# Test code
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

losser = SSIM(data_range=1.).cuda()
loss = losser(img1, img2).mean()

losser2 = MS_SSIM(data_range=1.).cuda()
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
