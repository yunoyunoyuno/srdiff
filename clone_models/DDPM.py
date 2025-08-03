import torch;
from clone_utils.utils import expand_axis_like;

class DDPM:
    def __init__(self, pred_x0_func, schedule):
        """ Sampler Abstract class

        Args:
            pred_x0_func: A function that predicts clean image x_0 given x_t (tensor shape (B, C, H, W)) and t (tensor shape (B)).
            schedule: A diffusion scheduler contains alpha and beta parameters.

        """
        self.pred_x0_func = pred_x0_func
        self.schedule = schedule

    # return x_(t-1)
    def update_step(self, x_t, t, context):
        """One step update

        Update x_t to x_{t-1} following DDPM update rule.

        Args:
            x_t (torch.tensor): An image at diffusion step t.
            t (torch.tensor): A diffusion timestep.

        Returns:
            x_mean (torch.tensor): The noiseless mean of the reverse process (not adding noise yet)
            x_update (torch.tensor): The updated image from a reverse process (mean + noise)

        """
        z = torch.randn_like(x_t)
        pred_x_0 = self.pred_x0_func(x_t, t, context) # the output of pred_x0_func is already clipped. You do not need to clip it anymore.

        # calculate the mean and std of p_\theta(x_{t-1} | x_t) where the mean is calculate from the equation above (in form of x_0).
        alpha_t = self.schedule.alpha[t];
        alpha_bar_t = self.schedule.alpha_cumprod[t];
        alpha_bar_prev_t = self.schedule.alpha_cumprod_prev[t];
        beta_t = self.schedule.beta[t];
        
        c1 = expand_axis_like(x_t,torch.sqrt(alpha_t)*(1-alpha_bar_prev_t)/(1-alpha_bar_t));
        c2 = expand_axis_like(pred_x_0,torch.sqrt(alpha_bar_prev_t)*(1-alpha_t)/(1-alpha_bar_t));
        x_mean = c1 * x_t + c2 * pred_x_0;

        # sample x_{t-1} using the reparameterization trick
        sigma_t = expand_axis_like(z,torch.sqrt((1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t));
        x_update = x_mean + sigma_t * z;
        
        return x_mean, x_update


    # return all images from (x_(T) to x_0)
    def sampling(self, x_T, context, return_all=False):
        """Sampling new image from prior sample

        Args:
            x_T (torch.tensor): A prior sample, sampling from Standard Normal Distribution.
            return_all (bool): If True, return every x_t for all t. Otherwise, only return x_0. Default to False.

        Returns:
            x_0 (torch.tensor): The generated images given prior samples, assuming x_0 is noiseless.

        """
        x_t = x_T
        T = self.schedule.T
        reverse_process = [x_T]
        for t in reversed(range(0, T)):
            vec_t = torch.ones(x_T.shape[0], dtype=int, device=x_T.device) * t
            x_mean, x_t = self.update_step(x_t, vec_t, context)
            # append x_t to reverse_process (Do not forget the last step.)
            reverse_process.append(x_t if t > 0 else x_mean)
        # if return_all is True then return reverse_process (list of x_t), otherwise return x_0
        
        reverse_process = torch.cat(reverse_process, dim=0) if return_all else reverse_process[-1]
        return reverse_process  # In the last step, we assume x_0 is noiseless.