import os
import time
import sys

import numpy as np

np.random.seed(101)

import torch

torch.manual_seed(101)

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MODEL_PATH = "./age_est_model.pt"
DATAPATH = './'


class Network(nn.Module):
    def __init__(self, labels):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 30, 5, padding=0)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=0)
        self.lin1 = nn.Linear(50 * 3 * 3 * 256, 200)
        self.lin2 = nn.Linear(200, 200)
        self.lin3 = nn.Linear(200, labels)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5)
        x = self.lin3(x)
        x = F.softmax(x)

        return x


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def main():
    print("loading npy files....")

    train_np = np.load(DATAPATH + 'ageTrain.npy')
    train_gt_np = np.load(DATAPATH + 'ageTrain_gt.npy')

    valid_np = np.load(DATAPATH + 'ageValid.npy')
    valid_gt_np = np.load(DATAPATH + 'ageValid_gt.npy')

    print("----------------------------------------------------")
    print("Number of training samples   : %d" % len(train_np))
    print("Number of validation samples : %d" % len(valid_np))
    print("----------------------------------------------------")
    print()

    EPOCHS = 1000
    LR = 1e-3
    REG = 1e-4

    labels = int(sys.argv[1:][0])

    model = Network(labels)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=REG)

    train_batch_size = 100
    valid_batch_size = 100

    train_batches = len(train_np) / train_batch_size
    valid_batches = len(valid_np) / valid_batch_size

    def training(epoch):
        def onehot_training():  # onehot encoder for training labels
            labels_onehot = torch.zeros(train_batch_size,
                                        labels)  # initialize labels_onehot with all zero
            for i in range(train_batch_size):  # loop through all images in batch
                labels_onehot[i][target_data[i]] = 1  # make index=1 for col=target label, rest 0
            return labels_onehot

        model.train()
        training_loss = 0
        total_correct = 0
        avg_time = 0
        for batch_id in range(int(train_batches)):
            start_t = time.time()

            input_data = train_np[batch_id * train_batch_size: (batch_id + 1) * train_batch_size]
            input_data = Variable(torch.from_numpy(input_data).float(), requires_grad=False)

            target_data = train_gt_np[batch_id * train_batch_size: (batch_id + 1) * train_batch_size]
            target_data = onehot_training()
            target_data = Variable(target_data, requires_grad=False)

            output = model(input_data)
            batch_loss = loss_func(output, target_data)
            training_loss += batch_loss.data[0]
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            end_t = time.time() - start_t
            avg_time = (avg_time * batch_id + end_t) / (batch_id + 1)
            epoch_left = (len(train_np) / train_batch_size - (batch_id + 1))
            time_left = avg_time * epoch_left

            print(
                '\rTrain itr: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime left: {:.2f}\titr left: {}\tAvg time per itr: {:.2f}'.
                    format(epoch, (batch_id + 1) * train_batch_size, len(train_np),
                           100.0 * (batch_id + 1) * train_batch_size / len(train_np),
                           batch_loss.data[0], time_left, epoch_left, avg_time), end='')

        average_training_loss = training_loss / (len(train_np) / train_batch_size)

        accuracy = total_correct / (len(train_np)) * 100

        print("\nAverage Training loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".
              format(average_training_loss, total_correct, len(train_np),
                     accuracy))

    def validation(epoch):
        def onehot_validation(target):  # onehot encoder for validation labels
            labels_onehot = torch.zeros(valid_batch_size, labels)  # initialize labels with all zeros
            for i in range(valid_batch_size):  # loop through all images in batch
                labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
            return labels_onehot

        model.eval()
        validation_loss = 0
        total_correct = 0

        for batch_id in range(int(valid_batches)):
            input_data = valid_np[batch_id * valid_batch_size: (batch_id + 1) * valid_batch_size]
            input_data = Variable(torch.from_numpy(input_data).float(), requires_grad=False)

            target_data = valid_gt_np[batch_id * valid_batch_size: (batch_id + 1) * valid_batch_size]
            onehot_target = onehot_validation(target_data)
            onehot_target = Variable(onehot_target, requires_grad=False)

            output = model(input_data)
            batch_loss = loss_func(output, onehot_target)
            validation_loss += batch_loss.data[0]

            value, index = torch.max(output.data, 1)
            target_data = torch.tensor(target_data).type(torch.LongTensor)

            for i in range(0, valid_batch_size):
                if index[i][0] == target_data[i]:
                    total_correct += 1

        average_validation_loss = validation_loss / (len(valid_np) / valid_batch_size)

        validation_accuracy = total_correct / (len(valid_np)) * 100

        print('\nValidation itr {}: Average loss: {:.6f} \t Accuracy: {}/{} ({:.2f}%)\n'.
              format(epoch, average_validation_loss, total_correct, len(valid_np),
                     validation_accuracy))

    for epoch in range(EPOCHS):
        start_time = time.time()
        training(epoch)
        end_time = time.time() - start_time
        print("Train Iteration: ", str(epoch + 1), ", Time Taken: ", end_time, "\n")
        validation(epoch)

        torch.save(model.state_dict(), MODEL_PATH)
        print("-> age estimation model is saved to %s" % MODEL_PATH)
    return


if __name__ == '__main__':
    main()