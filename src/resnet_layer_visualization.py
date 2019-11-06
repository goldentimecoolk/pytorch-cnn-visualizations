"""
Created on Sat Nov 18 23:12:08 2017
Updated on Wes Nov 06 17:00:00 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam, SGD
from torchvision import models
from collections import OrderedDict
from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        ##self.model[self.selected_layer].register_forward_hook(hook_function)
        self.model.layer4[2].conv3.register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (448, 448, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 61):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            ##for index, layer in enumerate(self.model):
            for name, layer in self.model.named_children():
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if name == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    cnn_layer = 'layer4' #resnet.layer4[2].conv3
    filter_pos = 5
    # Fully connected layer is not needed

    ##pretrained_model = models.vgg16(pretrained=True).features
    Model = models.resnet50(pretrained=True)
    Model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
    linear_in = Model.fc.in_features
    Model.fc = torch.nn.Linear(linear_in, 200)
    path = '/home/jsk/s/prcv/CNN/model/BASELINE-RESNET50.pkl'
    weight = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in weight.items():
        name = k[7:]  # remove "module."
        new_state_dict[name] = v
    Model.load_state_dict(new_state_dict,strict=True)
    pretrained_model = Model
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()
