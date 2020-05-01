import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

def main():
   
    input = get_args()
    
    checkpoint_1 = input.checkpoint
    cat_names = input.category_names
    gpu = input.gpu
    path_to_image = input.image_path
    num = input.top_k
   
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load(checkpoint_1)
    
    
    img = Image.open(path_to_image)
    image = processing(img)
    probs, classes = predict(path_to_image, model, num)
    check(image, path_to_image, model)
    
# Functions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()


class NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):

        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        i = 0
        j = len(hidden_layers)-1

        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1

        for each in hidden_layers:
            print(each)

        self.output = nn.Linear(hidden_layers[j], output_size)

    def forward(self, tensor):

        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
        tensor = self.output(tensor)

        return F.log_softmax(tensor, dim=1)
    
    
def load(x):
    
    checkpoint_2 = torch.load(x)
    model = getattr(torchvision.models, checkpoint_2['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = NN(checkpoint_2['input_size'],
                             checkpoint_2['output_size'],
                             checkpoint_2['hidden_layers'],
                             checkpoint_2['drop'])
    model.classifier = classifier

    
    model.classifier.load_state_dict(checkpoint_2['state_dict'])
    
    model.classifier.optimizer = checkpoint_2['optimizer']
    model.classifier.epochs = checkpoint_2['epochs']
    model.classifier.learning_rate = checkpoint_2['learning_rate']

    return model

def processing(image):
    
    w, h = image.size

    if w == h:
        size = 256, 256
    elif w > h:
        ratio = w/h
        size = 256*ratio, 256
    elif h > w:
        ratio = h/w
        size = 256, 256*ratio
        
    image.thumbnail(size, Image.ANTIALIAS)
    
    image = image.crop((size[0]//2 - 112, size[1]//2 - 112, size[0] + 112, size[1] - 112))
    
    
    img_array = np.array(image)
    np_image = img_array/255
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean)/std
    
    
    img = image.transpose((2, 0, 1))
    
    return img

def predict(image_path, model, num):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    image = processing(img)
    output = model(image)
    probs, indices = output.topk(topk)
    
    
    index_to_class = {val: key for key, val in cat_to_name.items()} #get class names from dict
    top_classes = [index_to_class[each] for each in indices]
    
    return probs, top_classes
    

def check(image, image_path, model):
    """
        Ouput a picture of the image and a graph representing its top 'k' class labels
    """
    probs, classes = predict(image_path, model)
    sb.countplot(y = classes, x = probs, color ='blue', ecolor='black', align='center')
    
    plt.show()
    ax.imsow(image)
    

    

if __name__ == "__main__":
    main()
