import torch
import numpy as np
from models import SocialLSTM, SocialModel
from loss import Gaussian2DLoss, SocialModelLoss
from data import Dataset
from random import shuffle
from torch.optim import Adam
from tqdm import tqdm

def main():
    # preparing
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    # load the data
    print("************* Loading Dataset ***************")
    # training_data = [np.random.rand(17, 20, 2) for _ in range(10)]
    dataset = Dataset()
    dataset.load_data('./data.h5')
    training_data, testing_data = dataset.get_train_validation_batch(20)

    print("************* Preparing Model ***************")
    # prepare a model
    model = SocialModel()

    # and the optimizer
    optimizer = Adam(model.parameters(), 1e-3)

    num_epochs = 100
    predict_length = 6

    print("************* Start Training ***************")
    for epoch in range(num_epochs):
        shuffle(training_data)
        total_loss = 0
        for batch in tqdm(training_data):
            batch = torch.from_numpy(batch).to(device).float()
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

            # updating step
            loss.backward()
            optimizer.step()

        print('Epoch {}: Loss: {}'.format(
            epoch + 1, total_loss / len(training_data)
        ))

if __name__ == '__main__':
    main()