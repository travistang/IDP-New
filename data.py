import pandas as pd
import numpy as np 
from sklearn import preprocessing
from random import shuffle
from tqdm import tqdm
import h5py

class Dataset():

    def __init__(self, dataset_path = './data/data_normalized.csv'):
        try:
            self.data = pd.read_csv(dataset_path)
        except:
            self.data = None

    def normalize_coordaintes(self):
        '''
            Given a dataset loaded as self.data, normalize X and Y column so that their values are in [-1, 1]
        '''
        self.data.X = (self.data.X - self.data.X.mean()) / (self.data.X.max() - self.data.X.min()) * 2
        self.data.Y = (self.data.Y - self.data.Y.mean()) / (self.data.Y.max() - self.data.Y.min()) * 2
        
        return self.data

    def get_train_validation_batch(self, seq_length, train_val_split = 0.8):
        '''
            Return Loaded data if it exists
        '''
        if self.training_data is not None and self.testing_data is not None:
            return self.training_data, self.testing_data

        '''
            Structure of the data: [n, seq_length, 2], where n is the number of vehicles in the particular time window
        '''
        print("************ get unique timestamps ***************")
        unique_timestamps = self.data.timestamp.unique()
        available_timestamps = unique_timestamps[unique_timestamps + seq_length < unique_timestamps.max()].tolist()

        shuffle(available_timestamps)
        
        print("************ splitting training / validation data ***************")
        split_idx = int(len(available_timestamps) * train_val_split)
        train_timestamp, test_timestamp = available_timestamps[:split_idx], available_timestamps[split_idx:]

        def get_trajectories_from_timestamp(start_ts):
            '''
                Small routine for sorting and filtering invalid trajectories starting from a given timestamp.
            '''
            # get all the timestamp between [ts, ts + seq_length]
            ts_list = list(range(start_ts, start_ts + seq_length))
            # get all records with timestamp within this range, and take just their coordinates and ID and ts.
            data_in_seq = self.data[self.data.timestamp.isin(ts_list)][['ID', 'X', 'Y', 'timestamp']]
            # group the record by ID, which should be like 
            # { ID_1: [[x1, y1], [x2, y2], ...], ID_2,: [...]}
            trajectories = []
            for _, group in data_in_seq.groupby(data_in_seq.ID):
                # if there are not as many record as the sequence length, it is invalid.
                if group.ID.count() != seq_length: continue
                # sort the group (all records from this particular ID) by ts, and filter out just the coordinates
                trajectories.append(
                    group.sort_values(by = 'timestamp')[['X', 'Y']].to_numpy()
                )

            # now this one should be of shape (n, seq_length, 2), where n is the number of unique cars seen in this particular timestamp.
            return np.array(trajectories) 
        
        print("************ aggregating data ***************")        
        # map the function to all the starting timestamps in training and testing data
        training_data = [get_trajectories_from_timestamp(start_ts) for start_ts in tqdm(train_timestamp)]
        testing_data  = [get_trajectories_from_timestamp(start_ts) for start_ts in tqdm(test_timestamp)]
        
        print("************ done ***************")
        return training_data, testing_data

    def save_data(self, training_data, validation_data = None, to_file = './data.h5'):
        with h5py.File(to_file, 'w') as f:
            group = f.create_group('training')
            
            [group.create_dataset('training-{}'.format(i), data = data) 
                for i, data in tqdm(enumerate(training_data), desc = 'saving training data...')]

            if validation_data:
                group = f.create_group('testing')
                [group.create_dataset('test-{}'.format(i), data = data) 
                    for i, data in tqdm(enumerate(validation_data), desc = 'saving validation data...')]
    
    def load_data(self, dataset_path):
        with h5py.File(dataset_path, 'r') as f:
            training_group = f['training']
            self.training_data = [
                training_group[dest].value
                for dest in training_group.keys()
            ]

            testing_group = f['testing']
            self.testing_data = [
                testing_group[dest].value
                for dest in testing_group.keys()
            ]
        
if __name__ == '__main__':
    dataset = Dataset()
    training_data, testing_data = dataset.get_train_validation_batch(20)
    dataset.save_data(training_data, testing_data)