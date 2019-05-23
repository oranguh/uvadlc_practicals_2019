import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.layers = OrderedDict()
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        self.layers["linear_0"] = nn.Linear(self.latent_dim, 128)
        self.layers["leakyrelu_0"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)

        self.layers["linear_1"] = nn.Linear(128, 256)
        self.layers["batchnorm_1"] = nn.BatchNorm1d(256)
        self.layers["leakyrelu_1"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)

        self.layers["linear_2"] = nn.Linear(256, 512)
        self.layers["batchnorm_2"] = nn.BatchNorm1d(512)
        self.layers["leakyrelu_2"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)

        self.layers["linear_3"] = nn.Linear(512, 1024)
        self.layers["batchnorm_3"] = nn.BatchNorm1d(1024)
        self.layers["leakyrelu_3"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)

        self.layers["linear_4"] = nn.Linear(1024, 784)
        self.layers["tanh_4"] = nn.Tanh()
        #   Output non-linearity ???

        self.classifier = nn.Sequential(self.layers)
        print("Created model : {}".format(self))
    def forward(self, z):

        # Generate images from z
        out = self.classifier(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = OrderedDict()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        self.layers["linear_0"] = nn.Linear(784, 512)
        self.layers["leakyrelu_0"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.layers["dropout_0"] = nn.Dropout(0.3)

        self.layers["linear_1"] = nn.Linear(512, 256)
        self.layers["leakyrelu_1"] = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.layers["dropout_1"] = nn.Dropout(0.3)

        self.layers["linear_2"] = nn.Linear(256, 1)
        self.layers["sigmoid_2"] = nn.Sigmoid()
        #   Output non-linearity ??

        self.classifier = nn.Sequential(self.layers)
        print("Created model : {}".format(self))

    def forward(self, img):
        # return discriminator score for img
        out = self.classifier(img)
        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args):

    criterion = nn.BCELoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    z_to_follow = np.random.normal(loc=0.0, scale=1.0, size=(100, args.latent_dim))
    z_to_follow = torch.from_numpy(z_to_follow).to(dtype=torch.float, device=device)

    generator_losses = []
    discriminator_losses = []
    generator_losses_mean = []
    discriminator_losses_mean = []

    discriminator.to(device)
    generator.to(device)

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # ones = torch.ones((imgs.shape[0],1), dtype=torch.float, device=device)
            zeros_ = torch.zeros((imgs.shape[0],1), dtype=torch.float, device=device)

            ones = torch.FloatTensor(imgs.shape[0],1).uniform_(0.7, 1.2).to(dtype=torch.float, device=device)
            zeros = torch.FloatTensor(imgs.shape[0],1).uniform_(0, 0.3).to(dtype=torch.float, device=device)
            answers = torch.cat((ones, zeros), dim=0)

            z = np.random.normal(loc=0.0, scale=1.0, size=(imgs.shape[0], args.latent_dim))
            z = torch.from_numpy(z).to(dtype=torch.float, device=device)
            generated_imgs = generator.forward(z)
            predictions_generated = discriminator.forward(generated_imgs)

            imgs = imgs.to(device)
            imgs = imgs.view(imgs.shape[0], -1)
            predictions_real = discriminator.forward(imgs)
            predictions = torch.cat((predictions_generated, predictions_real), dim=0)

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            discriminator_loss = criterion(predictions, answers)
            discriminator_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            generator_loss = criterion(predictions_generated, zeros_)
            generator_loss.backward()
            optimizer_G.step()

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:

                generator_losses_mean.append(np.mean(generator_losses))
                discriminator_losses_mean.append(np.mean(discriminator_losses))
                generator_losses = []
                discriminator_losses = []

                torch.save(generator.state_dict(), args.load_generator)

                with torch.no_grad():

                    if True:
                        path = 'images_gan/GAN_losses.png'
                        plt.plot(generator_losses_mean, label='generator loss')
                        plt.plot(discriminator_losses_mean, label='discriminator loss')
                        plt.title("Training losses GAN")
                        plt.xlabel("Epochs")
                        plt.ylabel("Loss")
                        plt.legend()
                        plt.savefig(path)
                        plt.close('all')

                    if True:
                        generated_imgs = generator.forward(z_to_follow)
                        to_show = generated_imgs.view(-1,1,28,28)
                        save_image(to_show,
                                   'images_gan/{}.png'.format(batches_done),
                                   nrow=10, normalize=True)
                    if True:
                        start_z = np.random.normal(loc=0.0, scale=1.0, size=(args.latent_dim))
                        end_z = np.random.normal(loc=0.0, scale=1.0, size=(args.latent_dim))
                        matr = np.linspace((start_z),(end_z),10)
                        matr = torch.from_numpy(matr).to(dtype=torch.float, device=device)
                        generated_imgs = generator.forward(matr)
                        to_show = generated_imgs.view(-1,1,28,28)
                        save_image(to_show,
                                   'images_gan/interpolation_{}.png'.format(batches_done),
                                   nrow=10, normalize=True)

                    print("epoch: {}".format(epoch))

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    # optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lr)
    # Start training



    if Path(args.load_generator).exists():
        generator.load_state_dict(torch.load(args.load_generator))

    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, args)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        # axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--load_generator', type=str, default="mnist_generator.pt",
                        help='Load generator give path')
    args = parser.parse_args()

    main()
