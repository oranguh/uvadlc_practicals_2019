import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid
from torchvision.utils import save_image
from scipy.special import logit, expit

def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    # dist = torch.distributions.normal.Normal(0, 1, validate_args=None)
    # logp = dist.log_prob(x)
    prior_logged = -0.5 * (x ** 2 + np.log(2 * np.pi))
    return prior_logged


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = np.random.normal(loc=0.0, scale=1.0, size=(size))
    sample = torch.from_numpy(sample).to(dtype=torch.float, device='cuda')

    # if torch.cuda.is_available():
    #     sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden)
            )
        self.translation = nn.Linear(n_hidden, c_in)
        self.scale = nn.Linear(n_hidden, c_in)

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        # the mask is used to make the determinant calculation easier. math magic
        leftover_mask = z * self.mask

        # you use the masked entry and fit it to the rest of your data
        temp = self.nn(leftover_mask)
        translation = self.translation(temp)
        scale = torch.tanh(self.scale(temp))

        """
            As computing the Jacobian determinant of the transformation
            is crucial to effectively train using this principle, this work exploits the simple observation that the
            determinant of a triangular matrix can be efficiently computed as the product of its diagonal terms.

            Since we still want to use all of the information we need to only calculate the ldj
            for the diagonal. But we want to push the information forward. Then the next coupling
            layer will do the other diagonal.

            literally copied the real NVP formula
        """

        if reverse:
            z = leftover_mask + (1 - self.mask) * ((z - translation) * scale.mul(-1).exp())

        else:
            # is it correct to see the translation as a kind of bias term?
            z = leftover_mask + (1 - self.mask) * (z * scale.exp() + translation)
            # since we are taking the log discrimants, we can simply keep summing the ldj
            # determinant of a triangular matrix can be efficiently computed as the product of its diagonal terms
            ldj += torch.sum((1 - self.mask) * scale, dim=1)

        return z, ldj

class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            # you make sure you do it with the reverse mask to approximate it
            # better? Or is it a good 1 on 1?
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        # add noise to your data to make sure your distribution is not degernate
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, z):
        """
        Given input, encode the input to z space. Also keep track of sum of log Jacobian determinant.
        """
        ldj = torch.zeros(z.size(0), device='cuda')

        z = self.dequantize(z)

        z, ldj = self.logit_normalize(z, ldj)
        z, ldj = self.flow.forward(z, ldj)


        # Compute log_pz and log_px per example
        # You first start with a simple normal distribution. You then
        # can make it complex by adding the sums of the determinants
        # log Jacobian determinant. Why do we sum the log instead of just multiply?
        # maybe numerical stability? So that you can sum instead?
        # from real NVP equation 3

        prior_ll = log_prior(z)
        # print(log_pz.shape, ldj.shape)
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1)# - np.log(256) * np.prod(z.size()[1:])

        log_px = prior_ll + ldj
        log_px = -log_px.mean()

        # log_px = log_pz.sum(dim=1) + ldj
        # log_prior_x = log_prior_z + sum(determinants)

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create
        log Jacobian determinant.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        z, ldj = self.flow.forward(z, ldj, reverse=True)
        z, ldj = self.logit_normalize(z, ldj, reverse=True)
        # we don't really care about ldj now do we.

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """
    bpds = []
    for batch in data:
        training, validation = batch

        loss = model.forward(training.cuda())
        # loss = loss.mul(-1).mean()
        # print(loss)

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0)
            optimizer.step()

        bpds.append(loss.item())

    # avg_bpd = logit(0.05 + (1-0.05) * (np.mean(bpds) / 256))
    avg_bpd =  (np.mean(bpds) / 784) * np.log(2)

    return avg_bpd.item()


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.ylim(0,5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        generated_imgs = model.sample(100)
        to_show = generated_imgs.view(-1,1,28,28)
        save_image(to_show,
                   'images_nfs/{}.png'.format(epoch),
                   nrow=10, normalize=True)


        save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')

    ARGS = parser.parse_args()

    main()
