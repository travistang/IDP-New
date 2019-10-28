import torch
from torch.distributions.multivariate_normal import MultivariateNormal
# import torch.nn.functional as F

def Gaussian2DLoss(target_coord, prediction_tensor):
    '''
        Compute the negative log likelihood of (x, y) given bivariate gaussian with params (mx, my, sx, sy, sxy),
        The PDF is given by k * (sx * sy - sxy ** 2) ** -0.5 * exp(-0.5 * (mx - x) * (inv )) 
        Assume batch of n points,
        target_coord: (n, 2), in each row, it is (x, y)
        prediction_tensor: (n, 5), in each row, it is (mx, my, sx, sy, sxy)
    '''
    mx, my, sx, sy, sxy = torch.split(prediction_tensor, 1, dim = 1) # each should be (n, 1)

    # so it would be a (n, 2, 2) matrix, each n is a covariance matrix of that node.    
    conv_mat = torch.stack([sx, sxy, sxy, sy], dim = -1).reshape(-1, 2, 2)
    # so the predicted convariance matrix is always positive semidefinite
    conv_mat = conv_mat @ torch.transpose(conv_mat, 1, 2)
    # mean vector: (n, 2)
    mean_vec = torch.stack([mx, my], dim = 1)[..., 0]

    # the bivariate pdf
    pdf = MultivariateNormal(mean_vec, conv_mat)

    # the negative log likelihood loss
    losses = (-pdf.log_prob(target_coord)).mean()

    return losses
    
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
    # initial_config[:, -1] /= 10
    # initial_config[:, 3] *= 
    # pdf_configs = Variable(initial_config, requires_grad = True)
    optimizer = Adam([pdf_configs], lr = 1e-2)
    # targets = torch.stack((
    #         torch.ones(n),
    #         torch.zeros(n)
    #     ), dim = 1)
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