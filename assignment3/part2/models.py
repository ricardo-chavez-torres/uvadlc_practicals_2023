################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-20
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, z_dim):
        """
        Convolutional Encoder network with Convolution and Linear layers, ReLU activations. The output layer
        uses a Fully connected layer to embed the representation to a latent code with z_dim dimension.
        Inputs:
            z_dim - Dimensionality of the latent code space.
        """
        super(ConvEncoder, self).__init__()
        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.z_dim = z_dim
        c_hid = 32
        num_input_channels = 1
        act_fn = nn.ReLU
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 1@28x28 => 32@14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32@14x14 => 64@7x7
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 64@7x7 => 64@4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, z_dim)
        )

        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Inputs:
            x - Input batch of Images. Shape: [B,C,H,W]
        Outputs:
            z - Output of latent codes [B, z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # x = None
        z = self.net(x)
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return z

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class ConvDecoder(nn.Module):
    def __init__(self, z_dim):
        """
        Convolutional Decoder network with linear and deconvolution layers and ReLU activations. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(ConvDecoder, self).__init__()
        # For an intial architecture, you can use the decoder of Tutorial 9. You can set the
        # output padding in the first transposed convolution to 0 to get 28x28 outputs.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        num_input_channels = 1
        self.num_input_channels = num_input_channels
        c_hid = 32
        act_fn = nn.GELU
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=0, padding=1, stride=2), # 4x4 => 7x7
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
            nn.Tanh() 
        )
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        recon_x = self.net(x)
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return recon_x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        """
        Discriminator network with linear layers and LeakyReLU activations.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(Discriminator, self).__init__()
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # As a default setup, we recommend 3 linear layers (512 for hidden units) with LeakyReLU activation functions (negative slope 0.2).
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            preds - Predictions whether a specific latent code is fake (<0) or real (>0). 
                    No sigmoid should be applied on the output. Shape: [B,1]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        preds = self.net(z)
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return preds

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class AdversarialAE(nn.Module):
    def __init__(self, z_dim=8):
        """
        Adversarial Autoencoder network with a Encoder, Decoder and Discriminator.
        Inputs:
              z_dim - Dimensionality of the latent code space. This is the number of neurons of the code layer
        """
        super(AdversarialAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(z_dim)
        self.decoder = ConvDecoder(z_dim)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z = self.encoder(x)
        recon_x = self.decoder(z)
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return recon_x, z

    def get_loss_autoencoder(self, x, recon_x, z_fake, lambda_=1):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
            recon_x - Reconstructed image of shape [B,C,H,W]
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
            lambda_ - The reconstruction coefficient (between 0 and 1).

        Outputs:
            recon_loss - The MSE reconstruction loss between actual input and its reconstructed version.
            gen_loss - The Generator loss for fake latent codes extracted from input.
            ae_loss - The combined adversarial and reconstruction loss for AAE
                lambda_ * reconstruction loss + (1 - lambda_) * adversarial loss
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        recon_loss = F.mse_loss(recon_x, x)
        discriminator_qz = self.discriminator(z_fake)

        gen_loss = F.binary_cross_entropy_with_logits(discriminator_qz, torch.ones_like(discriminator_qz)) 
        
        ae_loss = (1 - lambda_) * gen_loss +  lambda_ * recon_loss 

        logging_dict = {"gen_loss": gen_loss,
                        "recon_loss": recon_loss,
                        "ae_loss": ae_loss}
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return ae_loss, logging_dict

    def get_loss_discriminator(self,  z_fake):
        """
        Inputs:
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]

        Outputs:
            disc_loss - The discriminator loss for real and fake latent codes.
            logging_dict - A dictionary for logging the model performance by following keys:
                disc_loss - The discriminator loss for real and fake latent codes.
                loss_real - The discriminator loss for latent codes sampled from the standard Gaussian prior.
                loss_fake - The discriminator loss for latent codes extracted by encoder from input
                accuracy - The accuracy of the discriminator for both real and fake samples.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z_real = torch.randn_like(z_fake)

        real_out = self.discriminator(z_real)
        fake_out = self.discriminator(z_fake)

        loss_real = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))
        loss_fake = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))

        disc_loss = loss_real + loss_fake

        accuracy = ((real_out >= 0).sum() + (fake_out <0).sum()) / (z_fake.shape[0]*2)

        logging_dict = {"disc_loss": disc_loss,
                        "loss_real": loss_real,
                        "loss_fake": loss_fake,
                        "accuracy": accuracy}
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        return disc_loss, logging_dict

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random or conditioned images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        x = self.decoder(z)
        # raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return self.encoder.device


