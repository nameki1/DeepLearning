import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch

traindata = torchvision.datasets.MNIST(
    root='./dataset',
    train=True,download=True,
    transform=transforms.ToTensor()
)

data_loader = torch.utils.data.DataLoader(traindata,
                         batch_size=1,
                         shuffle=False)

data_iter = iter(data_loader)
for i in range(1,100):
    images, labels = data_iter.next()

    npimg = images[0].numpy()
    npimg = npimg.reshape((28, 28))
    plt.imshow(npimg, cmap='gray')
    filename = 'img/'+str(i)+'.png'
    plt.savefig(filename)



