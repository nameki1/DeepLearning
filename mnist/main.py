import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

batch_size = 8
epoch = 20

acc = []


# 学習データのダウンロード
traindata = torchvision.datasets.MNIST(
    root='./dataset',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
# 学習データを扱いやすい形に
trainloader = torch.utils.data.DataLoader(
    traindata,
    batch_size=batch_size,
    shuffle=True
)
# テストデータのダウンロード
testdata = torchvision.datasets.MNIST(
    root='./dataset',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# テストデータを扱いやすい形に
testloader = torch.utils.data.DataLoader(
    testdata,
    batch_size=batch_size,
    shuffle=True
)

# ネットワークモデルの設定
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,100)
        self.fc2 = nn.Linear(100,10)
        
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x
    
# モデルの初期化
net = Net()

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化アルゴリズム
optimizer = optim.SGD(net.parameters(), lr=0.01)


# 学習
def train():
    # 学習データ数(今回は60000)/batch_size 回だけ繰り返す
    for i, (inputs,labels) in enumerate(trainloader):
        # 計算情報の初期化
        optimizer.zero_grad()
        # ニューラルネットワークの計算
        output = net(inputs)
        # 損失関数を使って評価
        loss = criterion(output, labels)
        # どのパラメータをどれくらい調整するか計算する
        loss.backward()
        # 最適化アルゴリズムでパラメータを調整
        optimizer.step()

    
# テスト
def test():
    correct,total = 0,0

    # テストデータ数(今回は10000) / batch_size 回だけ繰り返す
    for i, (inputs,labels) in enumerate(testloader):
        # ニューラルネットワークの計算
        output = net(inputs)

        # ニューラルネットワークの出力を選択
        _, predicted = output.max(1)
        # 出力と実際の数字がいくつあっているかを計算
        correct += predicted.eq(labels).sum().item()

    # 100*正解した数 / テストデータ数 = 正答率
    print('Acc: %.2f%%' % (100*correct/10000))
    # この後結果のグラフを書くために正答率をリストに格納
    acc.append(100*correct/10000)
    

# main
def main():
    # まずはじめに学習していない状態でテストする
    print('\nEpoch: 0')
    test()
    # epoch回、学習とテストを繰り返す
    for e in range(1,epoch+1):
        print('\nEpoch: %d' % e)
        train()
        test()


    # 正答率の表示
    plt.plot(range(epoch+1),acc)
    plt.show()
    
if __name__ == '__main__':
    main()
