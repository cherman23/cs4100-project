from torch import nn
from torch import optim
from model import CNN

criterion = nn.CrossEntropyLoss()
model = CNN()

optimizer = optim.SGD(model.parameters(), lr=0.001) 