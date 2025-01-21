import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils import benchmark


print(f"Current working directory: {os.getcwd()}, "
      f"torch version: {torch.__version__}, "
      f"cuda version: {torch.version.cuda}, "
      f"cuDNN version: {torch.backends.cudnn.version()}, "
      f"available GPUs: {torch.cuda.device_count()}, "
      f"cuda device name: {torch.cuda.get_device_name()}")

typ=torch.float16
n= 1024*16
a=torch.randn(n,n).type(typ).cuda()
b=torch.randn(n,n).type(typ).cuda()
t=benchmark.Timer(stmt='a@b',globals={'a':a,'b':b})
x=t.timeit(50)
print(2*n**3/x.median/1e12)

