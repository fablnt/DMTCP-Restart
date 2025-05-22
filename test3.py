import time
import sys
import dmtcp
import torch

# Print all arguments
print("Arguments:", sys.argv)

# Initialize tensor on GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

# Initialize tensor
a = torch.tensor(0, dtype=torch.int32, device=device)
print("Initial tensor:", a)

# Access arguments
if len(sys.argv) > 1:
    print("First argument:", sys.argv[1])
    print("Second argument:", sys.argv[2] if len(sys.argv) > 2 else "No second argument")
else:
    print("No arguments passed")

# Main loop
try:
    while True:
        a = a + 1  # Increment on GPU (or CPU if no GPU)
        print("Counter:", a.item())
        time.sleep(1)
        
        if a.item() == 10:
            if device.type == "cuda":
                # Move to CPU for checkpointing
                a = a.to("cpu")
                torch.cuda.synchronize()  # Ensure all GPU operations are complete
                print("Moved tensor to CPU for checkpointing")
            
            # Perform checkpoint
            try:
                print("Attempting checkpoint...")
                dmtcp.checkpoint()
                print("Checkpoint completed")
            except Exception as e:
                print(f"Checkpoint failed: {e}")
            
            if device.type == "cuda":
                # Move back to GPU
                a = a.to(device)
                print("Moved tensor back to GPU")
except KeyboardInterrupt:
    print("Stopped by user")
