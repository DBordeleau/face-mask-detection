This was a two-phase project where I trained an AI to do image classification and make predictions about whether or not someone in a photo is wearing a face mask. A simple walkthrough of the project can be found below and a live demo of the deployed web app can be accessed at: 

The Jupyter notebook I used when training the model is included in the repo as "Main.ipynb"

# Training the model

Note: This is not the best way to go about this if you want the result to be the most accurate model you could train. There are so many pretrained models that you could start with that would produce significantly more accurate models than this one. I specifically went this route because I wanted to familiarize myself with the structure of a CNN and so I could learn how to train a CNN from scratch in PyTorch.

I trained the model using [this dataset I found on Kaggle.com](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection). The dataset contains 8928 images distributed equally into 3 folders "mask_without", "mask_with" and "mask_worn_incorrectly" which make up our prediction categories. 

The CNN is very basic and very similar to [the one NeuralNine uses in his image classification tutorial](https://www.youtube.com/watch?v=CtzfbUwrYGI). I only made changes to accommodate the structure of my dataset. The CNN is a very basic 3 layer convolutional neural net.

```
class MaskClassificationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Output: 32 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64 x 32 x 32
            nn.Dropout(0.25),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Output: 128 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128 x 16 x 16
            nn.Dropout(0.25),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 256 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output: 256 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 256 x 8 x 8
            nn.Dropout(0.5),

            # Flatten and Fully Connected Layers
            nn.Flatten(),
            nn.Linear(16384, 1024),  # Input size for Linear layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # Output: 3 classes
        )
```

To give the model the best chance at learning from the dataset I defined two sets of transformations to apply to the images before feeding them to the CNN. This is a common practice known as pre-processing. The first is a set of base transformations that gets applied to every image. The color jitter and random affine help to make the dataset a bit more diverse and I found they did make significant improvements to the model's ability to make predictions on novel data.

```
base_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(saturation=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

```

The second is a set of transformations that get applied only to the images of incorrectly worn masks. Early on in my training I found that the model was struggling with this class significantly more than any other and I thought altering the brightness and contrast of the images might help the model train on these images specifically.

```
incorrect_mask_augmentations = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
])
```

The actual training loop is very basic but I did make some changes in an attempt to improve the model. Some of these changes led to significant improvements. I've highlighted some of the more impactful changes below:

Increasing the weight of the incorrectly worn mask class to 3 when calculating loss resulted in significant improvements in the model's ability to distinguish this class from the with_mask class. I also tried 2.5:1:1 and 3.5:1:1 but 3 produced the most accurate model. It's worth noting that among all the training models I encountered while reading/studying for this project these weights stand out as pretty extreme. Still, they produced the best results.

```
class_weights = torch.tensor([3.0, 1.0, 1.0], device=device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

I was initially training the model with a learning rate of .001 for all epochs. Eventually I changed it so that the learning rate would adjust when the validation loss was plateauing. This made a significant difference when it came to overfitting, which I think is still a big problem prevalent in this model given it's relatively high test accuracy and low loss even on the first epoch.

```
learning_rate = 0.001
weight_decay = 1e-5 

# Initialize the optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler to reduce the learning rate when loss is plateauing
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
```

While these changes did significantly improve the model's ability to distinguish incorrectly worn masks from correctly worn masks, it still struggles. I am continuing to work on this project and one thing I hope to implement for the next version of the model is resizing. Currently all the images are 64 x 64 which is a big limitation when you consider the model relies on details like whether or not someone's nostrils are visible when it's trying to distinguish between image classes. This is not a very difficult change to implement, but I will have to make changes to the structure of the neural net mid-training to accommodate the resized images so I have some more learning to do.


# Getting Predictions from the model

I built an API for the model using Flask that would allow users to make requests for predictions. [This PyTorch article was quite helpful.](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) I quickly designed a front end using Tailwind and Flowbite. The webpage has a drop area where users can upload images and a button to submit a request to the API for a prediction. The API returns the prediction category and the confidence % of the prediction.

![](https://i.imgur.com/HQOcknB.png)

The Flask app in this repo is set up to use v2 of my model. You can set it to use v1 by modifying this line in app.py if you want to see how much the changes outlined in the training section above improved the model.

```
# change to "model_v1_scripted.pth" to see the differences my adjusted training model made
model = torch.jit.load('model_v2_scripted.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

# Conclusions

All in all this seems like quite a lot of work to end up with a mask classification mask that isn't very good, but I'm really happy with this project! At the outset of this I didn't have a concrete understanding of what a CNN was and I had never used PyTorch before. Now I understand the structure of a CNN and feel very comfortable using PyTorch to define neural net structures, load datasets, transform data, train prediction models and have even used it to make some simple visualizations like confusion matrices. I'm happy with the skills I acquired building this app and I will definitely continue to work on machine learning projects.

# Next/Improvements

While I continue to study machine learning I would like to make the following improvements to the project:

- I want the model to be even better at distinguishing between with_mask and mask_worn_incorrectly. It's still quite bad at this. I believe the improvements I introduced to the training model were only the beginning and that the model can still get significantly better with this dataset. The model has a very high initial rate of learning. The accuracy is high and the loss is low even on the first epoch. This suggests to me that the model is quickly learning basic distinctions but ultimately failing to learn about the nuanced differences that distinguish with_mask and mask_worn_incorrectly. I think I can make this better by changing the way I reduce the learning rate.

- I think it would be cool if the web app had the option to use the user's webcam feed and make live predictions about whether or not they're wearing a mask. 

- I'd like to add some visualizations to the front end. I think it would be awesome if the API also returned a visualization providing some context for why the AI predicted what it did and where its confidence stems from.