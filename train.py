import torch
import numpy as np
from models import SocialLSTM, SocialModel, VanillaLSTMModel
from loss import Gaussian2DLoss, SocialModelLoss
from data import Dataset
from random import shuffle, choice
from torch.optim import Adam
from tqdm import tqdm


import pandas as pd

def plot_inference(data, model, prediction_length, fname):
    import matplotlib.pyplot as plt
    from plot import get_normal, plot_normal, plot_tracks

    plt.clf()

    # the cumulative probability distribution of predictions in [-1, 1] x [-1, 1]
    cum_track = None
    for track in infer(
        data[:, :-prediction_length], 
        model, prediction_length).split(1, dim = 0):
        res, x, y = get_normal(track[0])
        cum_track = res if cum_track is None else res + cum_track

    # the accumulate distribution
    plt.contourf(x, y, cum_track)
    plot_tracks(data)
    plt.savefig(fname)

def generate_fake_data(num_trajectories, trajectory_length, prediction_length, min_step_length, max_step_length):
    data = []
    for i in range(num_trajectories):
        trajectory = []

        start_point = (np.random.rand(2) - 1) / 2 # start from somewhere at [-1, -0.5]
        trajectory.append(start_point)

        for ts in range(trajectory_length - 1):
            '''
                *----|-----*
                |--a-|
                |-----1----|

                    *-------|-----------*
                    |---c---|
                    |---------d---------|

                c / d = a => c = a * d
            '''
            slope = np.random.rand() * np.pi / 3 # some slope between (0, pi / 3): acute angle
            step_length = np.random.rand() * (max_step_length - min_step_length) + min_step_length
            point = trajectory[-1] # the previous point
            trajectory.append((
                point[0] + step_length * np.cos(slope), 
                point[1] + step_length * np.sin(slope),
            ))

        data.append(trajectory)

    training_data = np.array(data)
    testing_data  = np.array(data)

    return training_data, testing_data

def experiment(model, training_data, testing_data,
    hidden_size = 128, 
    embedding_size = 64, 
    num_epochs = 100,
    prediction_length = 10,

    random_rotation_angle = 40,

    model_name = 'social_lstm'):
    
    print('*********************** Experiment on {}, Hidden Size: {}, Embedding Size: {} ************************'.format(
        model_name, hidden_size, embedding_size
    ))

    # Common config
    lr = 1e-4

    optimizer = Adam(model.parameters(), lr)

    model_config_name = "{}_{}_{}".format(model_name, hidden_size, embedding_size)

    # preview prediction before train

    plot_inference(
        choice(training_data), 
        model, 
        prediction_length, 
        "{}_before_train.png".format(model_config_name))     
    
    print("******* Start Training ***********")

    training_loss, testing_loss = train_test(
        training_data, testing_data, 
        model, optimizer, 
        num_epochs, prediction_length
    )

    write_loss_to_csv(
        training_loss, testing_loss, 
        "{}.csv".format(model_config_name)
    )

    plot_inference(
        choice(training_data), 
        model, 
        prediction_length, 
        '{}_after_train.png'.format(model_config_name))

    # save weights
    torch.save(model.state_dict(), '{}.pth'.format(model_config_name))

def rotate_trajectories(data, random_rotation_angle):
    t = np.random.uniform(-random_rotation_angle, random_rotation_angle)
    # to radian
    t = t / 180 * np.pi
    rot_mat = np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t),  np.cos(t)]
    ])
    
    # rotate data 
    rotated_data = data @ rot_mat
    
    n, ts, _ = rotated_data.shape

    flattened_data = rotated_data.reshape(n * ts, 2)

    # get min max for rescaling back to [-1, 1]
    x_max, y_max = flattened_data.max(axis = 0)
    x_min, y_min = flattened_data.min(axis = 0)

    # re-interpolate coordinates in [-1, 1]
    if x_max > 1 or x_min < -1:
        rotated_data[..., 0] = 2 * (rotated_data[..., 0] - x_min) / (x_max - x_min) - 1
    if y_max > 1 or y_min < -1:
        rotated_data[..., 1] = 2 * (rotated_data[..., 1] - y_min) / (y_max - y_min) - 1

    return rotated_data

def main(args):

    # metadata for training
    trajectory_length = 20
    prediction_length = trajectory_length // 2

    print("************* Loading Dataset ***************")
    # training_data = [np.random.rand(17, 20, 2) for _ in range(10)]
    dataset = Dataset()
    dataset.load_data(args.dataset)
    training_data, testing_data = dataset.get_train_validation_batch(trajectory_length)

    # # reduce the size of training data..
    # training_data = training_data[:1000]
    # testing_data  = testing_data[:100]

    
    # experiment configs 
    experiment_embedding_size = [16, 32, 64, 128]
    experiment_hidden_size = [32, 64, 128, 256]

    for embedding_size in experiment_embedding_size:
        for hidden_size in experiment_hidden_size:

            social_model = SocialModel(
                hidden_size = hidden_size, 
                embedding_size = embedding_size)
            
            lstm_model = VanillaLSTMModel(hidden_size, embedding_size)

            experiment(
                social_model, training_data, testing_data,
                hidden_size = hidden_size,
                embedding_size = embedding_size,
                num_epochs = 100,
                model_name = 'social_lstm'
            )

            experiment(
                lstm_model, training_data, testing_data,
                hidden_size = hidden_size,
                embedding_size = embedding_size,
                num_epochs = 100,
                model_name = 'vanilla_lstm' 
            )

    print("done!")

def write_loss_to_csv(training_loss, testing_loss, csv_name):
    loss_data = {
        'training_loss': training_loss,
        'testing_loss': testing_loss,
    }

    pd.DataFrame(data = loss_data).to_csv(csv_name)

def train_test(training_data, testing_data, model, optimizer, num_epochs, predict_length):
    training_history = []
    testing_history  = []

    for epoch in range(num_epochs):
        # augmentation - rotating data

        random_rotation_angle = 270
        training_data = [
            rotate_trajectories(data, random_rotation_angle)
            for data in training_data
        ]

        testing_data = [
            rotate_trajectories(data, random_rotation_angle)
            for data in testing_data
        ]

        training_loss = train(training_data, model, optimizer, epoch, predict_length)
        testing_loss  = test(testing_data, model, predict_length)
        
        training_history.append(training_loss)
        testing_history.append(testing_loss)

        print('Epoch {}: Training loss: {}, Testing loss: {}'.format(
            epoch, training_loss, testing_loss
        ))

    return training_history, testing_history

def infer(data, model, predict_length):
    '''
        Treat the whole data as observation, infer `predict_length` steps
    '''
    print("************* Inference ***************")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    observation = torch.from_numpy(data).to(device).double()

    # pad preview
    pad = torch.ones(*observation.shape[:-1], 1).to(device).double()
    observation = torch.cat((observation, pad), axis = 2)

    # beforehand
    _, hs, cs = model(observation)

    # predict steps
    predicted, hs, cs = model(
        torch.zeros(observation.size(0), predict_length, 3).to(device), 
        (hs, cs))
        
    return predicted

def test(testing_data, model, predict_length):
    print("************* Start Testing ***************")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total_loss = 0
    total_batches = 0
    for batch in tqdm(testing_data, desc = "Testing on validation set..."):
        batch = torch.from_numpy(batch).to(device).double()

        observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
        # add one extra dimension indicating the sequence is beginning
        pad = torch.ones(*observation.shape[:-1], 1).to(device).double()
        observation = torch.cat((observation, pad), axis = 2)

        # beforehand
        _, hs, cs = model(observation)

        # predict steps
        predicted, hs, cs = model(
            torch.zeros(batch.size(0), predict_length, 3).to(device), 
            (hs, cs))
        
        loss = SocialModelLoss(predicted, target)
        num_batches = observation.size(0)
        total_loss += loss.item() * num_batches
        total_batches += num_batches

    return total_loss / total_batches

def train(training_data, model, optimizer, num_epochs, predict_length):
    '''
        Train on a list / iterator of trainin data.
        the shape of the training data should be of type
            [(num_trajectories, timestamps, 2)]
        num_trajectories is the number of agents seen in the scene
        timestamps is the number of steps to be seen AND predicted.
    '''
    print("************* Start Training ***************")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    shuffle(training_data)

    total_loss = 0
    total_batches = 0

    pbar = tqdm(training_data)
    for batch in pbar:

        optimizer.zero_grad()

        batch = torch.from_numpy(batch).to(device).double()

        observation, target = batch[:, :-(predict_length)], batch[:, -predict_length:]
        # add one extra dimension indicating the sequence is beginning
        pad = torch.ones(*observation.shape[:-1], 1).to(device).double()
        observation = torch.cat((observation, pad), axis = 2)

        # beforehand
        _, hs, cs = model(observation)

        # predict steps
        predicted, hs, cs = model(
            torch.zeros(batch.size(0), predict_length, 3).to(device), 
            (hs, cs))

        loss = SocialModelLoss(predicted, target)
        num_batches = observation.size(0)
        total_loss += loss.item() * num_batches
        total_batches += num_batches
        # compute gradient
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # updating step        
        optimizer.step()

        # update progress bar
        pbar.set_description('Epoch: {}, loss: {}'.format(num_epochs, total_loss / total_batches))

    return total_loss / total_batches
        
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inference', default = False, action = 'store_true')
    parser.add_argument('--dummy', default = False, action = 'store_true')
    parser.add_argument('--dataset', type = str, default = './data_transformed.h5')
    
    args = parser.parse_args()
    
    if args.inference:
        # do inference on random samples
        trajectory_length = 20

        dummy_model = VanillaLSTMModel(16, 16)
        # load data
        if not args.dummy:
            dataset = Dataset()
            dataset.load_data('./data_transformed.h5')
            training_data, testing_data = dataset.get_train_validation_batch(trajectory_length)
            # random sample 
            
            # load random model
            for i in range(20):
                plot_inference(
                    choice(training_data),
                    dummy_model,
                    trajectory_length // 2,
                    'sample_data_{}.png'.format(i + 1)
                )
        else:
            for i in range(20):
                training_data, testing_data = generate_fake_data(20, 20, 10, 0.05, 0.1)
                plot_inference(training_data, dummy_model, 10, "fake_{}.png".format(i))
    else:
        main(args)
