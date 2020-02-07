import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from threading import Thread
import numpy as np
from models import gaussian_prediction

def get_normal(predicted_tensor, existing_tensor = None):
    '''
        Plot the normal track of shape (ts, 5)
    '''
    x, y = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
    sample_points = np.stack([x, y], axis = -1) # (240, 240, 2)
    res  = None
    for ts in range(predicted_tensor.size(0)):
        out = predicted_tensor[ts].view(1, -1).detach()
        mean, cov = gaussian_prediction(out)
        
        mean = mean.detach().cpu().numpy()
        conv_mat = cov.detach().cpu().numpy()

        rv = multivariate_normal(mean[0], conv_mat[0])

        samples = rv.pdf(sample_points)
        
        res = samples if res is None else (res + samples)

    if existing_tensor is not None:
        res = res + existing_tensor if res is not None else existing_tensor

    return res, x, y
    
def plot_normal(predicted_tensor, existing_tensor = None):
    res, x, y = get_normal(predicted_tensor, existing_tensor = existing_tensor)
    plt.contourf(x, y, res)

def plot_tracks(tracks):
    num_vehicles = tracks.shape[0]
    num_steps    = tracks.shape[1]
    for v in range(num_vehicles):
        for ts in range(tracks.shape[1] - 2):
            plt.plot(
                tracks[v, ts: (ts + 2), 0], tracks[v, ts: (ts + 2), 1], 
                '--', 
                color = (1 - ts * (1 / num_steps), ts * (1 / num_steps), 0), 
                linewidth = 1.2)

if __name__ == '__main__':
    import torch
    batch_size = 7
    input_tensor = torch.randn(batch_size, 5)
    res, x, y = get_normal(input_tensor)
    print(res.max(), res.min())
    plt.contourf(x, y, res)
    plot_tracks(np.random.rand(3, 12, 2) * 1e-1)
    plt.savefig('multivariate.png')