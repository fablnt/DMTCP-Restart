import subprocess
import torch
import os
import time

CHECKPOINT_FILE = "loop_state.pt"

#def run_gpu_test_as_subprocess():
#    subprocess.run(["python3", "gpu_test_once.py"], check=True)

def save_checkpoint(i):
    torch.save({'i': i}, CHECKPOINT_FILE)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        return torch.load(CHECKPOINT_FILE)['i']
    return torch.rand(3, 3)

if __name__ == "__main__":
    print("Running test...")
    #run_gpu_test_as_subprocess()
    print("Now entering checkpointable loop...")

    j = 0
    i = load_checkpoint()

    while True:
        if j % 1000000 == 0:
            with open('output.txt', 'a') as f:
                f.write(str(i))
            print(j)
            save_checkpoint(i)
        j += 1


