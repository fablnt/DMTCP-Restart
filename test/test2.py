import time
from datetime import datetime
import sys
import dmtcp
import torch
import numpy as np
import gc 
 
# Stampa tutti gli argomenti
print("Argomenti:", sys.argv)

Tensore = torch.tensor(0, device='cpu')
print(Tensore)

if torch.accelerator.is_available():
    a = Tensore.to(torch.accelerator.current_accelerator())
#a=0
while True:
    a += 1
    print(a)
    time.sleep(1)
    if a == 10:
        Tensore = a.to('cpu') 
        del a
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        time.sleep(1)
        print(torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        dmtcp.checkpoint()
        a = Tensore.to(torch.accelerator.current_accelerator())


