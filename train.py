import torch
import numpy as np
from models import SocialLSTM, SocialModel, VanillaLSTMModel
from loss import Gaussian2DLoss, SocialModelLoss
from data import Dataset
from random import shuffle
from torch.optim import Adam
from tqdm import tqdm

from plot import plot_normal

import pandas as pd

def main():
    # load the data
    print("************* Loading Dataset ***************")
    # training_data = [np.random.rand(17, 20, 2) for _ in range(10)]
    dataset = Dataset()
    dataset.load_data('./data.h5')
    training_data, testing_data = dataset.get_train_validation_batch(20)

    # reduce the size of training data..
    training_data = training_data[:100]
    testing_data  = testing_data[:10]

    print("************* Preparing Model ***************")
    # prepare a model
    model = SocialModel()
    lstm_model = VanillaLSTMModel(128, 64)
    # and the optimizer
    lr = 7e-4
    optimizer = Adam(model.parameters(), lr)
    lstm_optimizer = Adam(lstm_model.parameters(), lr)

    num_epochs = 10
    predict_length = 6

    # train the Social LSTM model
    # social_train_losses, social_test_losses = train_test(
    #     training_data, testing_data, 
    #     model, optimizer, num_epochs, predict_length)
    
    # write_loss_to_csv(
    #     social_train_losses, social_test_losses, 
    #     './social_loss.csv')

    lstm_train_losses, lstm_test_losses = train_test(
        training_data, testing_data, 
        lstm_model, lstm_optimizer, num_epochs, predict_length)
    
    write_loss_to_csv(
        lstm_train_losses, lstm_test_losses, 
        './lstm_loss.csv')
    
    # run once more for the test data to plot it.
    display_data = testing_data[:1]
    predicted = infer(display_data, lstm_model, 4)
    # draw the first track
    plot_normal(predicted[0])

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
        training_loss = train(training_data, model, optimizer, num_epochs, predict_length)
        testing_loss  = test(testing_data, model, predict_length)
        
        training_history.append(training_loss)
        testing_history.append(testing_loss)

        print('Epoch {}: Training loss: {}, Testing loss: {}'.format(
            epoch, training_loss, testing_loss
        ))

    return training_history, testing_history

def infer(data, model, predict_length):
    print("************* Inference ***************")
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for batch in tqdm(data, desc = "Inferencing..."):
        batch = torch.from_numpy(batch).to(device).double()
        # print(batch.shape)
        observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
        # print('target size', target.shape)
        # beforehand
        _, hs, cs = model(observation)

        # predict steps
        predicted, hs, cs = model(
            torch.zeros(batch.size(0), predict_length, 2), 
            (hs, cs))

    return predicted

def test(testing_data, model, predict_length):
    print("************* Start Testing ***************")
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    total_loss = 0
    for batch in tqdm(testing_data, desc = "Testing on validation set..."):
        batch = torch.from_numpy(batch).to(device).double()
        # print(batch.shape)
        observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
        # print('target size', target.shape)
        # beforehand
        _, hs, cs = model(observation)

        # predict steps
        predicted, hs, cs = model(
            torch.zeros(batch.size(0), predict_length, 2), 
            (hs, cs))
        
        loss = SocialModelLoss(predicted, target)
        total_loss += loss.item()

    return total_loss / len(testing_data)

def train(training_data, model, optimizer, num_epochs, predict_length):
    print("************* Start Training ***************")
    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    shuffle(training_data)
    total_loss = 0
    for batch in tqdm(training_data):
        batch = torch.from_numpy(batch).to(device).double()
        # print(batch.shape)
        observation, target = batch[:, :-(predict_length + 1)], batch[:, -predict_length:]
        # print('target size', target.shape)
        # beforehand
        _, hs, cs = model(observation)

        # predict steps
        predicted, hs, cs = model(
            torch.zeros(batch.size(0), predict_length, 2), 
            (hs, cs))

        # print('predicted', predicted.shape)
        loss = SocialModelLoss(predicted, target)
        total_loss += loss.item()

        # compute gradient
        loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # updating step        
        optimizer.step()

    return total_loss / len(training_data)
        
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    main()