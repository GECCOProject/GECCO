import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from util import new_model, load_model, load_checkpoint, load_mnist
from tqdm import tqdm
import os
from model import Model

BATCH_SIZE = 64
source = 'new'
#source = 'checkpoint'
checkpoint_step = 0
checkpoint_path = './models/gnn-checkpoint-{}.pt'.format(checkpoint_step)


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

trainloader, testloader = load_mnist()

images_val, labels_val = next(iter(testloader))
# images_val = torch.stack(images_val).to(device).float()
images_val = images_val.to(device).float()
labels_val = labels_val.to(device)

old_epoch = 0

if source == 'checkpoint':
    print('Load from checkpoint')
    model, optimizer, old_epoch, epoch_loss = load_checkpoint(device, checkpoint_path)
elif source == 'new':
    print('Create new model')
    model = new_model(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
else:
    assert 0
    
lambd = 1e-5

criterion = nn.CrossEntropyLoss()
num_epochs = 1000

total_time = 0

p_bar = tqdm(total=len(trainloader))
SIZE_DATA = len(trainloader) * BATCH_SIZE
best_loss = float('inf')
for epoch in tqdm(range(old_epoch + 1, num_epochs+1)):   
    epoch_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        # data: batch of 64 * [image, label]
        images, labels = data
        images = images.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, train=True)
        loss = F.log_softmax(criterion(outputs, labels))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        p_bar.update(1)
        
    # scheduler.step()
    epoch_loss /= len(trainloader)
    

# test
    model.eval()
    with torch.no_grad():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        outputs = model(images_val, train=False)
        ender.record()
        torch.cuda.synchronize()
        loss = F.log_softmax(criterion(outputs, labels_val))
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels_val).sum().item()
        val_accuracy = correct / labels_val.size(0)        
        latency = starter.elapsed_time(ender)
        total_time += latency
    
    
    p_bar.reset()
    latency /= len(trainloader)
    
    if epoch == 1:
        try:
            os.remove('train.log')
        except OSError:
            pass
        
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}, Latency: {:.4f}'.format(epoch, epoch_loss, loss, val_accuracy, latency)
    f = open('train.log', 'a')
    f.write(log + '\n')
    f.close()
    if epoch % 10 == 0:
        checkpoint_path = './models/gnn-checkpoint-{}.pt'.format(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_loss': epoch_loss
            }, checkpoint_path)
        print("Saved checkpoint number {}".format(epoch))
        
model_path = "./models/model.pt"
torch.save(model, model_path)

print('Finished Training')



