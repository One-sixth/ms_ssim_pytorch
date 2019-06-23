import torch
import sys
sys.path.append('../')

from no1_ms_ssim_lizhengwei1992_MS_SSIM_pytorch import MS_SSIM as MS_SSIM1
from no3_ssim_VainF_pytorch_msssim import MS_SSIM as MS_SSIM3
from ssim import MS_SSIM as MS_SSIM4


def test_speed(losser):
    a = torch.randint(0, 255, size=(20, 3, 256, 256), dtype=torch.float32).cuda() / 255.
    b = a * 0.5
    a.requires_grad = True
    b.requires_grad = True

    start_record = torch.cuda.Event(enable_timing=True)
    end_record = torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()
    start_record.record()
    for _ in range(500):
        loss = losser(a, b).mean()
        loss.backward()
    end_record.record()
    end_time = time.perf_counter()

    torch.cuda.synchronize()

    print('cuda time', start_record.elapsed_time(end_record))
    print('perf_counter time', end_time - start_time)


if __name__ == '__main__':
    print('Performance Testing MS_SSIM')
    print()
    import time
    losser1 = MS_SSIM1(size_average=False, max_val=1.).cuda()
    losser3 = MS_SSIM3(win_size=11, win_sigma=1.5, data_range=1., size_average=False, channel=3).cuda()
    losser4 = MS_SSIM4(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()

    print('testing losser1')
    test_speed(losser1)
    print()

    print('testing losser3')
    test_speed(losser3)
    print()

    print('testing losser4')
    test_speed(losser4)
    print()
