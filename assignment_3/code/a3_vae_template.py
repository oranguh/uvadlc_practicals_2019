import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import OrderedDict
from datasets.bmnist import bmnist
from torchvision.utils import save_image

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.hidden = nn.Linear(784, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)

        print("Created model : {}".format(self))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        hidden = torch.tanh(self.hidden(input.view(input.shape[0],-1)))
        mean = self.mu(hidden)
        std = self.std(hidden)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.layers = OrderedDict()
        self.layers["linear_0"] = nn.Linear(z_dim, hidden_dim)
        self.layers["tanh_0"] = nn.Tanh()
        self.layers["linear_1"] = nn.Linear(hidden_dim, 784)
        self.layers["sigmoid_1"] = nn.Sigmoid()

        self.decode = nn.Sequential(self.layers)

        print("Created model : {}".format(self))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decode(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim).to(device)
        self.decoder = Decoder(hidden_dim, z_dim).to(device)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean, variance_log = self.encoder.forward(input)

        self.mean = mean
        """
            STD is actually log variance?
        """
        # print(mean)
        std = torch.exp(variance_log*0.5)
        zeros = torch.zeros(mean.shape).to(self.device)
        noise = torch.normal(zeros, std).to(self.device)
        # noise = torch.randn_like(std)
        z = mean + noise.mul(std)

        """
            z should be the mean (MLE) plus some noise (variation)
        """
        # print(z)
        output = self.decoder.forward(z)
        # print(output.shape)
        # print(asdasd)
        if True:
            to_show = output[:100].view(-1,1,28,28)
            save_image(to_show,
                       'images/VAE_{}.png'.format(0),
                       nrow=10, normalize=True)
        # plt.matshow(input[0].view(28,28).cpu().detach().numpy())
        # plt.show()
        # print(asdasd)
        """
            Kullback–Leibler divergence
        """
        # ????
        # The loss for the modelling the latent space (generative part)
        kl = -0.5 * torch.sum(1 + variance_log - mean**2 - variance_log.exp())

        # the loss for reconstruction (the autoencoder part)

        # using magic numbers is bad bad but whatever
        # recon_loss = nn.BCELoss(output, input.view(input.shape[0], -1), reduction='sum')
        # Should reconstruction be sum or mean? I can understand that summing would make things more stringent for reconstruction
        recon_loss = F.binary_cross_entropy(output, input.view(input.shape[0], -1), reduction='sum')
        # print("recon: {}, kl: {}".format(recon_loss.item(), kl.item()))
        # the KL loss is always about 20% of the recon_loss.... I wonder if it should be higher?
        # print(kl.item())
        average_negative_elbo = kl*5 + recon_loss

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = None, None

        with torch.no_grad():

            # We repeat the mean vector for as many samples as we want
            # The shape will be (n_samples, length_of_latent)
            to_bernoullio = self.mean.repeat(n_samples, 1)
            # We then sample using those means, note means must be between
            # 0 and 1 for this to work. Returns binary values.
            z = torch.bernoulli(to_bernoullio)

            # push through these values through decoder
            sampled_ims = self.decoder.forward(z)
            # unflatten the vector into an image (black and white)
            sampled_ims = sampled_ims.view(-1, 1, 28, 28)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    # make list of all batches, then average at end
    average_epoch_elbo = []

    # dataloader data already gives you batches, usually size 128, but can be smaller
    for batch in data:
        batch = batch.to(device)
        elbo_batch = model.forward(batch)
        # print(asdasds)
        average_epoch_elbo.append(elbo_batch.item())
        if model.training:
            optimizer.zero_grad()
            elbo_batch.backward()
            optimizer.step()

    return np.mean(average_epoch_elbo)


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer, device)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
