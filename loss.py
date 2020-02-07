import torch
from torch.distributions.multivariate_normal import MultivariateNormal
# import torch.nn.functional as F
import numpy as np
from models import gaussian_prediction

def Gaussian2DLoss(target_coord, prediction_tensor, beta = 5, alpha = 1):
    '''
        Compute the negative log likelihood of (x, y) given bivariate gaussian with params (mx, my, sx, sy, sxy),
        The PDF is given by k * (sx * sy - sxy ** 2) ** -0.5 * exp(-0.5 * (mx - x) * (inv )) 
        Assume batch of n points,
        target_coord: (n, 2), in each row, it is (x, y)
        prediction_tensor: (n, 5), in each row, it is (mx, my, sx, sy, sxy)
    '''

    # get the modified prediction of the gaussian
    mean_vec, conv_mat =  gaussian_prediction(prediction_tensor)

    # the bivariate pdf
    pdf = MultivariateNormal(mean_vec, conv_mat)

    # the negative log likelihood loss

    raw_probs = torch.exp(pdf.log_prob(target_coord))
    # NLL loss
    losses = (-pdf.log_prob(target_coord))
    # breakpoint()
    final_loss, _ = torch.min(torch.stack((
        losses,
        (beta - raw_probs * ((beta + np.log(alpha)) / alpha)).abs()
    ), dim = -1), dim = -1)

    final_loss = torch.mean(final_loss)
    return final_loss
    
def SocialModelLoss(pred, target):
    return torch.mean(
        torch.stack([
            Gaussian2DLoss(target[i], pred[i]) 
            for i in range(pred.size(0))
        ])
    )

if __name__ == '__main__':
    '''
        Try optimizing through the loss functions with toy examples
    '''
    from torch.autograd import Variable
    from torch.optim import Adam, SGD
    import matplotlib.pyplot as plt
    from plot import plot_normal

    n = 5
    pdf_configs = torch.randn(n, 5).double().requires_grad_(True)

    optimizer = Adam([pdf_configs], lr = 1e-2)

    targets = torch.randn(n, 2)
    
    print('before training...')
    plot_normal(pdf_configs)
    plt.savefig('before_train.png')
    plt.clf()

    for i in range(1000):
        optimizer.zero_grad()
        losses = Gaussian2DLoss(targets, pdf_configs)
        if i % 10 == 0:
            print('Epoch {}; Loss: {}'.format(i + 1, losses))
        losses.backward()
        optimizer.step()
    print('after training...')
    plot_normal(pdf_configs)
    plt.savefig('after_train.png')
    plt.clf()