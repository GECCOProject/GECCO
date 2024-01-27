import torch
import numpy as np
import torch_geometric

from matplotlib.image import imsave
from time import time

from util import load_images, setup_model, load_checkpoint, load_mnist

checkpoint_path = "./models/gnn-checkpoint-60.pt"
torch.cuda.is_available()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

_, testloader = load_mnist()
# _, testloader = load_original(device)

def main(checkpoint_path, verbose=0):
    model, _, _, _ = load_checkpoint(device, checkpoint_path)
    # model = torch.load(checkpoint_path)
    model.eval()

    num_correct = 0
    num_test = 0
    total_time = 0.0
    count = 0
    total_imgs = 0
    temp_latency = 0.0
    if verbose:
        log = np.array([])
    for i, data in enumerate(testloader):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        images, labels = data
        # data: batch of 64 * [image, label]
        images = images.to(device)
        # images = torch.stack(images).to(device).float()
        labels = labels.to(device)
        
        outputs = model(images, train=False)
        predictions = outputs.max(1, keepdim=True)[1].squeeze()
        ender.record()
        torch.cuda.synchronize()
        
        curr_time = starter.elapsed_time(ender)
        total_time += curr_time
        total_imgs += labels.size()[0]
        num_correct += torch.sum(predictions == labels).item()
        num_test += labels.size()[0]
        count += 1
        
        temp_latency += curr_time / labels.size()[0]
        
        # Record throughputs
        
        if verbose:
            log_batch = (predictions==labels).cpu().detach().numpy()
            log = np.concatenate([log, log_batch])
            
    latency = temp_latency / count
    throughput = total_imgs / total_time
    if verbose:
        np.savetxt(checkpoint_path[:-3]+'-results.txt', log)
    print("Accuracy: {} / {} = {}".format(num_correct, num_test, num_correct/num_test))
    print("Latency: ", latency)
    print("Throughput: ", throughput)

if __name__ == '__main__':
    # checkpoint_path = "./models/gnn-checkpoint-280.pt"
    print(checkpoint_path)
    main(checkpoint_path, 1)

