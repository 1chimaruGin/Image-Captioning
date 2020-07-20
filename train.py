import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from loader import get_loader
from model import CNNtoRNN

def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = get_loader(
        root_folder = 'flickr8k/images/',
        annotation_file = 'flickr8k/captions.txt',
        transform = transform,
        num_workers = 2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    writer = SummaryWriter('logs/flickr')
    step = 0

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load('my_ckpt.pth.tar'), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        print_examples(model, device, epoch)
        if save_model:
            checkpoint = {
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'step' : step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions)[:-1]

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar('loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train()