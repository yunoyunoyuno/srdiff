import torch;
import torch.nn as nn;
from clone_utils.utils import expand_axis_like;
from typing import Literal;

class NoiseScheduler:
    def __init__(self, T, mode : Literal["linear","cosine"] = "linear"):
        """ Noise Scheduler Abstract Class

        Args:
            T (int): A maximum number of diffusion timestep.

        """
        self.T = T;
        self.mode = mode;
        self.init_alpha_beta();

    def init_alpha_beta(self):
        """ Initialize alpha and beta parameters based on the scheduler. """
        t = torch.linspace(start = 1,end=self.T,steps=self.T,dtype=torch.int) #[T,]
        if self.mode == "linear":
            self.beta = 1e-4*(1-(t-1)/(self.T-1)) + 0.02*((t-1)/(self.T-1)); #[T,]
            self.alpha = 1-self.beta; # [B,]
            self.alpha_cumprod = torch.cumprod(self.alpha,dim = -1); #[T,]
            self.alpha_cumprod_prev = torch.roll(self.alpha_cumprod,shifts=1); #[T,]
            self.alpha_cumprod_prev[0] = 1; 
            
        elif self.mode == "cosine":
            term = (t-1)/(self.T-1); #[T,]
            ft = torch.cos((term + 0.008) / 1.008 * (torch.pi / 2))**2 #[T,]
            self.alpha_cumprod = ft/(ft[0] + 1e-6);
            self.alpha_cumprod_prev = torch.roll(self.alpha_cumprod,shifts=1); #[T,]
            self.alpha_cumprod_prev[0] = 1;
            self.alpha = self.alpha_cumprod/self.alpha_cumprod_prev; #[T,]
            self.beta = torch.min(1-self.alpha,torch.tensor(0.999)); #[T,]
        else:
            raise NotImplementedError

    def _mean(self, x_0, t):
        """ Mean of p(x_t | x_0, t)

        Args:
            x_0 (Tensor): Clean images (Shape: (B, C, H, W))
            t (Tensor): Diffusion time-step (Shape: (B))

        Returns:
            Mean of p(x_t | x_0, t) (Shape: (B, C, H, W))

        """
        alpha_cumprod_t = expand_axis_like(x_0,self.alpha_cumprod[t]); #[1,] -> [1,1,1,1]
        x_mean = (alpha_cumprod_t).sqrt()* x_0; #[B,C,H,W] * [1,1,1,1] -> [B,C,H,W]
        return x_mean;

    def _std(self, t):
        """ Standard deviation of p(x_t | x_0, t)

        Args:
            t (Tensor): Diffusion time-step (Shape: (B))

        Returns:
            Standard deviation of p(x_t | x_0, t) (Shape: (B))

        """
        std = torch.sqrt(1-self.alpha_cumprod[t]); # [B,]
        return std

    def marginal_prob(self, x_0, t):
        """ Marginal probability p(x_t | x_0, t)"""
        return self._mean(x_0, t), self._std(t) #[B,C,H,W], [B,]

    def sample_marginal_prob(self, x_0, t, noise=None):
        """ Sample x_t from p(x_t | x_0, t)

        Args:
            x_0 (Tensor): Clean images (Shape: (B, C, H, W))
            t (Tensor): Diffusion time-step (Shape: (B))
            noise (Tensor): A gaussian noise to be used in the reparameterization trick.
                            If it is None, noise is sample from standard normal. Default to None
        Returns:
            x_t (Shape: (B, C, H, W))

        """
        mean, std = self.marginal_prob(
            x_0, t
        )  # Compute mean and std of the marginal prob.
        
        if(noise is None):
            noise = torch.randn_like(x_0)

        std = expand_axis_like(x_0,std); #[B,]->[B,1,1,1]
        x_t = mean + std*noise; #[B,C,H,W]+[B,1,1,1]->[B,C,H,W];
        return x_t

    def prior_sampling(self, resolution, batch_size=1, num_channels=3):
        """ Sampling Gaussian noise

        Args:
            resolution (Tuple[int]): A tuple of integer indicates width and height of the image.
            batch_size (int, optional): A number of noise to be generated.
            num_channels (int, optional): A number of channels in the images.

        Returns:
            The sampling noises sample from Gaussian distribution.

        """
        return torch.randn(batch_size, num_channels, *resolution) #[B,C,H,W]

    def to(self, *args, **kwargs):
        """ Store the parameters on the given devices (eg. cpu, cuda) """
        self.alpha = self.alpha.to(*args, **kwargs)
        self.beta = self.beta.to(*args, **kwargs)
        self.alpha_cumprod = self.alpha_cumprod.to(*args, **kwargs)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(*args, **kwargs)
        return self