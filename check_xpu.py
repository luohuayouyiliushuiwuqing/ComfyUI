import torch
print(torch.__version__)           # 应 >= 2.5.0
print(torch.xpu.is_available())    # 应返回 True
print(torch.xpu.device_count())    # 应 >= 1
print(torch.xpu.get_device_name(0)) # 应显示 "Intel(R) Arc(TM) 140T"