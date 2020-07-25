# Udacity-dcgan-face-generator

# Face Generation
In this project, I used generative adversarial networks to generate new images of faces that look as realistic as possible!

 ---
### Datasets
 - [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 This dataset contains over 200,000 ccelebrity images with annotations. 
 - Or can be Download and unzip in the local pc.
 
 ---
 ### Pre-processed Data
 Udacity team already done some sort of pre-processing for me. My part is **transform** this data and create a **DataLoader**
 
 ```python
 
 def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    #Tensor Transform
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    #put in ImageFolder and the data loader later
    train_dataset = datasets.ImageFolder(data_dir,transform=transform)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,
                              shuffle=True)
    
    return train_loader
    
 ```
 - I choose 128 as my batch size as in the [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), they say:
 > All models were trained with mini-batch stochastic gradient descent (SGD) with
a mini-batch size of 128.
---
**Scale the images**
---
As the output of a `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. 

 ```python
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min_ran, max_ran = feature_range
    return x * (max_ran - min_ran) + min_ran
 ```
 ---
# Define the Model Architectrue 
A GAN is comprised of two adversarial networks, a discriminator and a generator.
   - According to the guidelines proposed by the author in [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf)
    for stable Deep Convolutional GANs
      - Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
      - Use batchnorm in both the generator and the discriminator
      - Remove fully connected hidden layers for deeper architectures.
      - Use ReLU activation in generator for all layers except for the output, which uses Tanh.
      - Use LeakyReLU activation in the discriminator for all layers.
## Discriminator Discriminator
- Main Goal : input 32*32*3(RGB) tensor images output the value that determine the a given image is real or fake
- Define Helper Function that creates a convolutional layer, with optional batch normalization.

 ```python
 def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)
 ```
 ## Discriminator Model:
 ```python
 class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        #32*32*3 -> 16*16*128
        #don't apply batch norm for first layer
        self.conv1 = conv(in_channels=3,out_channels=conv_dim,kernel_size=4,batch_norm=False)
        #16*16*128 -> 8*8*256
        self.conv2 = conv(in_channels=conv_dim,out_channels=conv_dim*2,kernel_size=4)
        #8*8*256 -> 4*4*512
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4,kernel_size=4)
        # fully connected layer output is fake or real
        self.fcl = nn.Linear(conv_dim*4*4*4,1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        #all leaky relu slope was set to 0.2 by default
        #con1 
        x = F.leaky_relu(self.conv1(x))
        #con2
        x = F.leaky_relu(self.conv2(x))
        #con3
        x = F.leaky_relu(self.conv3(x))
        #flatten the images
        x = x.view(-1,self.conv_dim*4*4*4)
        #output
        out = self.fcl(x)
        return out
 ```
 ## Generator Discriminator
- Main Goal : inputs to the generator are vectors of some length `z_size` output`32x32x3`(RGB) tensor images 
- Define Helper Function that creates a deconvolutional layer, with optional batch normalization.
 ### helper deconv function 
 ```python
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    ## TODO: Complete this function
    ## create a sequence of transpose + optional batch norm layers
    layers = []
    Transpose2d = nn.ConvTranspose2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    # append conv layer
    layers.append(Transpose2d)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers) 
  ```
  ## Generator Model:
  ```python
  class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        #need from 4*4*512 to 32*32*3
        # complete init function
        self.conv_dim = conv_dim
        self.fcl = nn.Linear(z_size,conv_dim*4*4*4)
        #deconv 512*4*4 -> 256*8*8
        self.dcon1 = deconv(in_channels=conv_dim*4,out_channels=conv_dim*2,kernel_size=4)
        #deconv 256*8*8 -> 128*16*16
        self.dcon2 = deconv(in_channels=conv_dim*2,out_channels=conv_dim,kernel_size=4)
        #deconv 128*16*16 -> 3*32*32
        self.dcon3 = deconv(in_channels=conv_dim,out_channels=3,kernel_size=4,batch_norm=False)
        
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fcl(x)
        x = x.view(-1,self.conv_dim*4,4,4)
        x = F.relu(self.dcon1(x))
        x = F.relu(self.dcon2(x))
        out = F.tanh(self.dcon3(x))
        return out
  ```
---
## Inital Weight:
   > All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.
    
---
## Discriminator and Generator Losses  
### Discriminator Losses
> * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`. 
* Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.

### Generator Loss
The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to *think* its generated images are *real*.

  ```python
  def real_loss(D_out):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    #calculate loss 
    loss = criterion(D_out.squeeze(),labels)
    return loss
  ```
 ---
 ## Optimizers
 Define optimizers for models with appropriate hyperparameters.
  - The author propsed using learning rate 0.0002 is good as 0.001 is too high
  - Beta suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped
stabilize training.
  ```python
  import torch.optim as optim
  # params
  lr = 0.0002
  beta1= 0.5
  beta2= 0.9
  # Create optimizers for the discriminator D and generator G
  d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
  g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
 ```
