import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super(Network, self).__init__()
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])        
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        for each in self.hidden_layers:
            x = self.dropout(F.relu(each(x)))

        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
        
def train(model, trainloader, testloader, criterion, optimizer, epochs):
    ''' Train the neural network using a image dataset.
    
        Arguments
        ---------
        model: network model created using Network Class
        trainloader: iterator for image training
        testloader: iterator for image test
        criterion: NN Loss function
        optimizer: optimizer used to update the gradients
        epochs: integer, number of epochs used for training the model
        
    '''
    
    train_losses, test_losses = [], []
    
    for e in range(epochs):
        running_loss = 0
        loader = tqdm(trainloader, total=len(trainloader))
        for batch_idx, (train_images, train_labels) in enumerate(loader):
            train_images, train_labels = Variable(train_images), Variable(train_labels)
        
            optimizer.zero_grad()
            log_ps = model(train_images)
            loss = criterion(log_ps, train_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            test_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                model.eval()
                
                for test_images, test_labels in testloader:
                    log_ps = model(test_images)
                    test_loss += criterion(log_ps, test_labels)
                    ps = torch.exp(log_ps)
                    
                    top_p, top_class = ps.topk(1,dim=1)
                    equals = top_class == test_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
            model.train()
            
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))    
            
    return train_losses, test_losses
            
            
            
def plot_losses(train_l, test_l):
    plt.figure(figsize=(10,8))
    plt.plot(train_l, label='Training loss')
    plt.plot(test_l, label='Validation loss')
    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Values')
    plt.title('Classifier with Dropout')
    
    
def save_checkpoint(model, filepath):
    checkpoint = {'input_size': model.hidden_layers[0].in_features,
                  'output_size': model.output.out_features,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)    
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model    