import torch
import numpy as np
from models import SocialLSTM, SocialModel, VanillaLSTMModel
from loss import Gaussian2DLoss, SocialModelLoss
from data import Dataset
from reporter import Reporter
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
    hidden_size, embedding_size, 
    args,
    model_name = 'social_lstm'):
    
    print('*********************** Experiment on {}, Hidden Size: {}, Embedding Size: {} ************************'.format(
        model_name, hidden_size, embedding_size
    ))

    optimizer = Adam(model.parameters(), args.lr)

    model_config_name = "{}_{}_{}".format(model_name, hidden_size, embedding_size)

    # preview prediction before train
    ploting_sample = choice(training_data)
    if args.random_rotation_angle is not None:
        ploting_sample = rotate_trajectories(ploting_sample, args.random_rotation_angle)
    plot_inference(
        ploting_sample, 
        model, 
        args.prediction_length, 
        "{}_before_train.png".format(model_config_name))     
    
    training_loss, testing_loss = train_test(
        training_data, testing_data, 
        model, optimizer, args, model_name = model_name)

    print ("*************** Experiment Complete. Recording results ***************")
    write_loss_to_csv(
        training_loss, testing_loss, 
        "{}.csv".format(model_config_name)
    )

    plot_inference(
        ploting_sample, 
        model, 
        args.prediction_length, 
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

'''
    Main routine for training the models
'''
def main(args):

    # metadata for training
    trajectory_length = 20
    prediction_length = trajectory_length // 2

    print("************* Loading Dataset ***************")
    dataset = Dataset()
    dataset.load_data(args.dataset)
    training_data, testing_data = dataset.get_train_validation_batch(trajectory_length)

    # # reduce the size of training data..
    if args.truncated:
        print("Using truncated data")
        training_data = training_data[:10]
        testing_data  = testing_data[:10]


    # experiment configs 
    # experiment_embedding_size = [64]
    # experiment_hidden_size = [128]
    experiment_embedding_size = args.embedding_size
    experiment_hidden_size = args.hidden_size

    for embedding_size in experiment_embedding_size:
        for hidden_size in experiment_hidden_size:

            social_model = SocialModel(
                hidden_size = hidden_size, 
                embedding_size = embedding_size)
            
            experiment(
                social_model, training_data, testing_data,
                hidden_size, embedding_size,
                args,
                model_name = 'social_lstm'
            )

            # lstm_model = VanillaLSTMModel(hidden_size, embedding_size)
            # experiment(
            #     lstm_model, training_data, testing_data,
            #     hidden_size = hidden_size,
            #     embedding_size = embedding_size,
            #     num_epochs = 100,
            #     lr = args.lr,
            #     model_name = 'vanilla_lstm',
            # )

    print("done!")

def write_loss_to_csv(training_loss, testing_loss, csv_name):
    loss_data = {
        'training_loss': training_loss,
        'testing_loss': testing_loss,
    }

    pd.DataFrame(data = loss_data).to_csv(csv_name)

def train_test(training_data, testing_data, model, optimizer, args, **kwargs):
    training_history = []
    testing_history  = []

    for epoch in range(args.epochs):

        training_loss = train(training_data, model, optimizer, args, epoch, **kwargs)
        testing_loss  = test(testing_data, model, args, epoch, **kwargs)
        
        training_history.append(training_loss)
        testing_history.append(testing_loss)

        print('Epoch {} / {}: Training loss: {}, Testing loss: {}'.format(
            epoch, args.epochs, training_loss, testing_loss
        ))

    return training_history, testing_history

def infer(data, model, predict_length):
    '''
        Treat the whole data as observation, infer `predict_length` steps
    '''
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

def test(testing_data, model, args, epoch, **kwargs):
    print("************* Start Testing ***************")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    predict_length          = args.prediction_length
    random_rotation_angle   = args.random_rotation_angle
    reporter                = Reporter("ws://localhost:8080") if args.remote_monitor else None

    total_loss = 0
    total_batches = 0
    for batch_id, batch in enumerate(
        tqdm(testing_data, desc = "Testing on validation set...")
    ):
        if random_rotation_angle is not None:
            batch = rotate_trajectories(batch, random_rotation_angle)
        
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
        
        if reporter:
            reporter.report(
                model_name = kwargs['model_name'] + '_test', 
                loss = loss.item(), 
                batch_id = batch_id, epoch = epoch)

        num_batches = observation.size(0)
        total_loss += loss.item() * num_batches
        total_batches += num_batches

    return total_loss / total_batches

def train(training_data, model, optimizer, args, epoch):
    '''
        Train on a list / iterator of trainin data.
        the shape of the training data should be of type
            [(num_trajectories, timestamps, 2)]
        num_trajectories is the number of agents seen in the scene
        timestamps is the number of steps to be seen AND predicted.
    '''
    print("************* Start Training ***************")
    
    # destructure args
    num_epochs              = args.epochs
    predict_length          = args.prediction_length
    batch_size              = args.batch_size
    random_rotation_angle   = args.random_rotation_angle

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    shuffle(training_data)

    total_loss = 0
    total_num_vehicles = 0

    pbar = tqdm(range(0, len(training_data), batch_size))

    reporter = Reporter("ws://localhost:8080") if args.remote_monitor else None

    for start_id in pbar:
        batch_data = training_data[start_id: start_id + batch_size]
        # aggregate the batch loss here
        batch_loss = []
        for batch_id, batch in enumerate(batch_data):
            # augment if needed
            if random_rotation_angle is not None:
                batch = rotate_trajectories(batch, random_rotation_angle)
        
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
            num_vehicles = observation.size(0)
            total_loss += loss.item() * num_vehicles
            total_num_vehicles += num_vehicles
            loss.backward()

            if reporter:
                reporter.report(
                    model_name = "social lstm",
                    loss = loss.item(),
                    batch_id = start_id + batch_id, 
                    epoch = epoch,  
                )

        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # updating step        
        optimizer.step()

        # update progress bar
        pbar.set_description('Epoch: {} / {}, loss: {:.3f}'.format(epoch + 1, num_epochs, total_loss / total_num_vehicles))

    return total_loss / total_num_vehicles
        
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title = "mode")
    
    inference_parser = subparsers.add_parser('inference')
    inference_parser.set_defaults(which = 'inference')
    inference_parser.add_argument('--model_path', type = str, required = True)
    inference_parser.add_argument('--dummy', default = False, action = 'store_true')
    inference_parser.add_argument('--embedding_size', type = int, required = True)
    inference_parser.add_argument('--hidden_size', type = int, required = True)
    inference_parser.add_argument('--trajectory_length', type = int, default = 20)
    inference_parser.add_argument('--prediction_length', type = int, default = 10)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(which = 'train')
    train_parser.add_argument('--model_path', type = str, help = "The weights for model to finetune on")
    train_parser.add_argument('--dataset', type = str, default = './data_transformed.h5')
    train_parser.add_argument('--random_rotation_angle', type = float, default = 270)
    train_parser.add_argument('--epochs', type = int, default = 100)
    train_parser.add_argument('--truncated', action = 'store_true')
    train_parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
    train_parser.add_argument('--batch_size', type = int, default = 16)
    train_parser.add_argument('--remote_monitor', action = 'store_true', help = 'Use Remote Monitor')
    train_parser.add_argument('--trajectory_length', type = int, default = 20)
    train_parser.add_argument('--prediction_length', type = int, default = 10)
    train_parser.add_argument('--embedding_size', type = int, required = True, nargs = '+')
    train_parser.add_argument('--hidden_size', type = int, required = True, nargs = '+')
    
    args = parser.parse_args()

    if args.which is 'inference':
        # do inference on random samples
        trajectory_length = args.trajectory_length
        prediction_length = args.prediction_length

        model = SocialModel(args.embedding_size, args.hidden_size)
        with open(args.model_path, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
        # load data
        if not args.dummy:
            dataset = Dataset()
            dataset.load_data('./data_transformed.h5')
            training_data, testing_data = dataset.get_train_validation_batch(trajectory_length)
            # random sample 
            
            # load random model
            for i in tqdm(range(20), desc = "Inferring..."):
                plot_inference(
                    choice(training_data),
                    model,
                    prediction_length,
                    'sample_data_{}.png'.format(i + 1)
                )
        else:
            for i in range(20):
                training_data, testing_data = generate_fake_data(20, 20, 10, 0.05, 0.1)
                plot_inference(training_data, model, args.prediction_length, "fake_{}.png".format(i))
    else:
        main(args)
