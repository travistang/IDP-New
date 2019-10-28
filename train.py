import torch
import numpy as np
from models import SocialLSTM, SocialModel, VanillaLSTMModel
from loss import Gaussian2DLoss, SocialModelLoss
from data import Dataset
from random import shuffle
from torch.optim import Adam
from tqdm import tqdm

from plot import plot_normal, plot_tracks

import pandas as pd

def main():
    import matplotlib.pyplot as plt
    # load the data
    print("************* Loading Dataset ***************")
    # training_data = [np.random.rand(17, 20, 2) for _ in range(10)]
    # dataset = Dataset()
    # dataset.load_data('./data.h5')
    # training_data, testing_data = dataset.get_train_validation_batch(20)

    # # reduce the size of training data..
    # training_data = training_data[:100]
    # testing_data  = testing_data[:10]
    
    # create some fake trajectories
    # random starting point, constant, positive slope
    num_trajectories = 3
    trajectory_length = 20
    min_step_length, max_step_length = 0.05, 0.1
    
    data = []
    for i in range(num_trajectories):
        trajectory = []

        start_point = (np.random.rand(2) - 1) / 2 # start from somewhere at [-1, -0.5]
        trajectory.append(start_point)

        for _ in range(trajectory_length - 1):
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
                point[1] + step_length * np.sin(slope)
            ))

        data.append(trajectory)

    training_data = np.array(data)
    testing_data  = np.array(data)

    plot_tracks(training_data)
    plt.savefig('tracks.png')

    print("************* Preparing Model ***************")
    # prepare a model
    model = SocialModel()
    lstm_model = VanillaLSTMModel(128, 64)

    # and the optimizer
    lr = 1e-4
    optimizer = Adam(model.parameters(), lr)
    lstm_optimizer = Adam(lstm_model.parameters(), lr)

    num_epochs = 10
    predict_length = 6

    # preview of the inference of untrained models.
    plt.clf()
    plot_normal(infer(training_data, lstm_model, trajectory_length // 2))
    plt.savefig('lstm_predict_before_train.png')
    # train the Social LSTM model
    social_train_losses, social_test_losses = train_test(
        training_data, testing_data, 
        model, optimizer, num_epochs, predict_length)
    
    # write_loss_to_csv(
    #     social_train_losses, social_test_losses, 
    #     './social_loss.csv')
    

    lstm_train_losses, lstm_test_losses = train_test(
        training_data, testing_data, 
        lstm_model, lstm_optimizer, num_epochs, predict_length)
    
    write_loss_to_csv(
        lstm_train_losses, lstm_test_losses, 
        './lstm_loss.csv')
    
    predicted = infer(training_data, lstm_model, num_trajectories // 2)

    # draw the prediction
    plt.clf()
    plot_normal(predicted)
    plt.savefig('lstm_predict_after_train.png')

    print("done!")

    print('Comparison: Social LSTM loss: {}, Vanilla LSTM loss: {}'.format(np.mean(social_test_losses), np.mean(lstm_test_losses)))

    # save model
    torch.save(model.state_dict(), 'social_lstm.pth')
    torch.save(lstm_model.state_dict(), 'lstm.pth')


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
        training_loss = train(training_data, model, optimizer, epoch, predict_length)
        testing_loss  = test(testing_data, model, predict_length)
        
        training_history.append(training_loss)
        testing_history.append(testing_loss)

        print('Epoch {}: Training loss: {}, Testing loss: {}'.format(
            epoch, training_loss, testing_loss
        ))

    return training_history, testing_history

def infer(data, model, predict_length):
    print("************* Inference ***************")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    batch = torch.from_numpy(data).to(device).double()
    # print(batch.shape)
    observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
    
    # beforehand
    _, hs, cs = model(observation)

    # predict steps
    predicted, hs, cs = model(
        torch.zeros(batch.size(0), predict_length, 3), 
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
        # print(batch.shape)
        observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
        # add one extra dimension indicating the sequence is beginning
        pad = torch.ones(*observation.shape[:-1], 1).to(device).double()
        observation = torch.cat((observation, pad), axis = 2)
        # print('target size', target.shape)
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
    print("************* Start Training ***************")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('training on {}'.format(device))
    model.to(device)

    shuffle(training_data)

    total_loss = 0
    total_batches = 0

    pbar = tqdm(training_data)
    for batch in pbar:
        batch = torch.from_numpy(batch).to(device).double()
        # print(batch.shape)
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

        # print('predicted', predicted.shape)
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
    main()
