import torch
import torch.nn
import torchvision
import smnist_models as models
import numpy as np
import os
import getopt
import tools

import sys
sys.path.append("..")

import snn

################################################################
# CLI
################################################################

OPT_NONE = "none"
OPT_COMPILE = "compile"
OPT_TORCHSCRIPT = "torchscript"

GRAD_FGI = "fgi"
GRAD_BACK = "back"

opts, args = getopt.getopt(
    sys.argv[1:], "oigr", ["opt=","insize=","grad=","run="]
)

cfg_opt = OPT_NONE
cfg_insize = int(1)
cfg_grad = GRAD_BACK

for opt, arg in opts:
    if opt in ("-o", "--opt"):
        cfg_opt = arg
    elif opt in ("-i", "--insize"):
        cfg_insize = int(arg)
    elif opt in ("-g", "--grad"):
        cfg_grad = arg
    elif opt in ("-r", "--run"):
        cfg_run = int(arg)

################################################################
# General settings
################################################################

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

if torch.cuda.is_available():
    device = torch.device(DEVICE_CUDA)
    pin_memory = False
    num_workers = 1
    timer = tools.CUDATimer()
else:
    pin_memory = False
    num_workers = 0
    device = torch.device(DEVICE_CPU)

################################################################
# Data loading and preparation
################################################################
input_size = cfg_insize
sequence_length = int((28 * 28) / input_size)
num_classes = 10
train_batch_size = 128
test_batch_size = 128

train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# Only use a multiple of train_batch_size for training
# for a more accurate time measurements.
train_dataset_size = int(len(train_dataset) / train_batch_size) * train_batch_size
train_dataset, _ = torch.utils.data.random_split(
    train_dataset, [train_dataset_size, len(train_dataset) - train_dataset_size]
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True
)

def smnist_transform_input_batch(
    tensor: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    input_size: int
):
    tensor = tensor.to(device=device).view(batch_size, sequence_length, input_size)
    tensor = tensor.permute(1, 0, 2)
    return tensor


################################################################
# Model setup
################################################################
hidden_size = 128

print("device :", device)
print("opt    :", cfg_opt)
print("insize :", cfg_insize)
print("seqlen :", sequence_length)
print("grad   :", cfg_grad)
print("run    :", cfg_run)
print()

torch.set_float32_matmul_precision('high')

if cfg_grad == GRAD_FGI:
    model = snn.models.ALIFFGISNN(
        input_size, hidden_size, num_classes
    ).to(device)
else:
    model = snn.models.ALIFSNN(
        input_size, hidden_size, num_classes
    ).to(device)

if (cfg_opt == OPT_COMPILE):
    model = torch.compile(model, fullgraph=True, mode="max-autotune")
elif (cfg_opt == OPT_TORCHSCRIPT):
    model = torch.jit.script(model)


################################################################
# Setup experiment (optimizer etc.)
################################################################

criterion = torch.nn.CrossEntropyLoss()

optimizer_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr, amsgrad=True)

# Number of iterations per epoch
total_steps = len(train_loader)

epochs_num = 100
iterations_warmup = 10
print_every = 10

################################################################
# Training loop
################################################################

# Go into train mode.
model.train()

forward_pass_times = list()
backward_pass_times = list()

iterations = 0
max_iterations = 1000

timer = tools.CUDATimer()

abort = False

for epoch in range(epochs_num):

    # Perform training epoch
    for i, (inputs, targets) in enumerate(train_loader):

        # Reshape inputs in [sequence_length, batch_size, data_size].
        current_batch_size = len(inputs)

        inputs = smnist_transform_input_batch(
            tensor=inputs.to(device=device),
            sequence_length=sequence_length,
            batch_size=current_batch_size,
            input_size=input_size
        )

        target = targets.to(device=device)

        optimizer.zero_grad()

        # -------------------------
        # Measure forward pass time
        # -------------------------
        timer.reset()
        ##
        outputs = model(inputs)
        ##
        forward_time = timer.time()

        # Loss calculatino
        loss = criterion(outputs.mean(dim=0), target)
        loss_value = loss.item()

        # -------------------------
        # Measure forward pass time
        # -------------------------
        timer.reset()
        ##
        loss.backward()
        ##
        backward_time = timer.time()

        forward_pass_times.append(forward_time)
        backward_pass_times.append(backward_time)

        optimizer.step()

        if i % print_every == 0:
            print("Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f} | Forward time mean: {:.6f} | Backward time mean: {:.6f}".format(
                epoch + 1, epochs_num, i + 1, total_steps, loss_value, forward_time, backward_time), flush=True
            )

        iterations += 1

        if (iterations - iterations_warmup) >= max_iterations:
            abort = True
            break

    if abort:
        break

################################################################
# Logging
################################################################

results_dir = "results"
filename = results_dir + "/" "smnist_{}_{}_{}_{:03d}.csv".format(input_size, cfg_opt, cfg_grad, cfg_run)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

f = open(filename, "w")
for i in range(iterations):
    f.write(str(forward_pass_times[i]) + ", " + str(backward_pass_times[i]) + "\n")

f.close()
