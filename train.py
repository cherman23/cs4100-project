from torch import nn
from torch import optim
from model import CNN

criterion = nn.CrossEntropyLoss()

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.001) 

def train(model_name, classes, cycles):
    for _ in cycles:
        # Import data from data processing
        data = {}
        

        scores = model(data)
        loss = criterion(scores,classes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        nn.save(model.state_dict(),model_name + '.pth')