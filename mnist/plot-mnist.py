import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import random

# データのダウンロード
traindata = torchvision.datasets.MNIST(
    root='./dataset',              
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# データの表示
num = 5
fig, ax = plt.subplots(num,num)
for i in range(num):
    for j in range(num):
        l = random.randint(0,60000)
        ax[i,j].imshow(traindata[l][0].view(-1,28),cmap='gray')
        ax[i,j].axis("off")

plt.savefig("img.png")
plt.show()
