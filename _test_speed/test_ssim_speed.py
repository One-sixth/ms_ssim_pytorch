import torch
import sys
sys.path.append('../')

from no2_ssim_Po_Hsun_Su_pytorch_ssim import SSIM as SSIM2
from no3_ssim_VainF_pytorch_msssim import SSIM as SSIM3
from ssim import SSIM as SSIM4
from no5_pipa_ssim import SSIM as SSIM5


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
    print('Performance Testing SSIM')
    print()
    import time
    losser2 = SSIM2(window_size=11, size_average=False).cuda()
    losser3 = SSIM3(win_size=11, win_sigma=1.5, data_range=1., size_average=False, channel=3).cuda()
    losser4 = SSIM4(window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False).cuda()
    losser5 = SSIM5(window_size=11, value_range=1., n_channels=3, reduction='none').cuda()

    print('testing losser2')
    test_speed(losser2)
    print()

    print('testing losser3')
    test_speed(losser3)
    print()

    print('testing losser4')
    test_speed(losser4)
    print()

    print('testing losser5')
    test_speed(losser5)
    print()
