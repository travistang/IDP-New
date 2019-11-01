import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from threading import Thread
import numpy as np

def get_normal(predicted_tensor, existing_tensor = None):
    '''
        Plot the normal track of shape (ts, 5)
    '''
    x, y = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
    sample_points = np.stack([x, y], axis = -1) # (240, 240, 2)
    res  = None
    for ts in range(predicted_tensor.size(0)):
        mx, my, sx, sy, sxy = predicted_tensor[ts].detach().numpy().tolist()
        
        conv_mat =  np.array([[sx, sxy], [sxy, sy]])
        conv_mat = conv_mat @ conv_mat.T

        rv = multivariate_normal([mx, my], conv_mat)

        samples = rv.pdf(sample_points)
        
        res = samples if res is None else (res + samples)

    if existing_tensor is not None:
        res = res + existing_tensor if res is not None else existing_tensor

    return res, x, y
    
def plot_normal(predicted_tensor, existing_tensor = None):
    res, x, y = get_normal(predicted_tensor, existing_tensor = existing_tensor)
    plt.contourf(x, y, res)

def plot_tracks(tracks):
    for ts in range(tracks.shape[0]):
        plt.plot(tracks[ts, :, 0], tracks[ts, :, 1], 'ro--', linewidth = 1)

if __name__ == '__main__':
    import torch
    batch_size = 7
    # means  = torch.randn(batch_size, 2)
    # sx, sy = torch.split(torch.randn(batch_size, 2), 1, dim = 1) 
    # sxy    = torch.randn(batch_size, 1)
    # input_tensor = torch.cat((means, sx, sy, sxy), dim = 1)
    input_tensor = torch.randn(batch_size, 5)
    res, x, y = get_normal(input_tensor)
    print(res.max(), res.min())
    plt.contourf(x, y, res)
    plot_tracks(np.random.rand(3, 12, 2) * 1e-1)
    plt.savefig('multivariate.png')