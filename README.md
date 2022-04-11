# PyTorch-Tutorials
PyTorch - Deep learning framework devoloped by facebook research AI

For python environment install requirement libraries to run PyTorch dependencies (Linux-64-ARM)


    pip install -r requirements.txt

For Conda environment


    conda install -r requirements.txt

## Introduction to PyTorch

PyTorch is an open source python deep learning framework, developed primarily by Facebook that has been gaining momentum
recently. It provides the Graphics Processing Unit (GPU), an accelerated multidimensional array (or tensor) operation, and computational graphs, which we can be used to build
neural networks.


## Fashion - MNIST
### Transforming and augmenting images

Transforms are common image transformations available in the torchvision.
pytorch provides the ability to transform images to tensors 

#### Dataset Description:
Fashion mnist is a dataset of Zalando's image article it consist of 60k training samples and 10k testing samples. it has 10 classes and 28x28 pixel grayscale images 

#### Kaggle description of Fashion mnist:

Content

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

    To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27. The pixel is located on row i and column j of a 28 x 28 matrix. 


Labels

Each training and test example is assigned to one of the following labels:

    |--------------------------| 
    | labels    Class Names    |
    | ------    -----------    |
    | 0         T-shirt/top    |
    | 1         Trouser        |
    | 2         Pullover       | 
    | 3         Dress          |
    | 4         Coat           |
    | 5         Sandal         |
    | 6         Shirt          |
    | 7         Sneaker        |
    | 8         Bag            |
    | 9         Ankle boot     |
    |--------------------------|

