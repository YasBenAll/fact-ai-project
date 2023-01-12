import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy import genfromtxt
from dataset_loader import MyDataset
import create_data
from minimax_optimization import compute_weight_with_fairness_no_label, compute_weight_with_fairness_with_label
batch_size = 35000
learning_rate = 0.25
num_epochs = 10
filename = 'adult_norm.csv'
numpy_data = genfromtxt(filename, delimiter=',')
#numpy_data = numpy_data[np.argsort(numpy_data[:,5])]
print(numpy_data.shape)




index = batch_size
feature_size = 36
temp = numpy_data[:]
# train_data = numpy_data[:, 1:feature_size][:index]
# train_label = numpy_data[:, -1][:index]
#train_data, train_label, lower, upper, Sen, cluster = unique_data.weighted_data_eduation_with_unlabel(numpy_data)
train_data, train_label, lower, upper, Sen, cluster = create_data.weighted_adult_data_kmeans_wihtout_unlabel(numpy_data)

print(len(train_data))

numpy_data = temp[:]
test_data = numpy_data[:, 1:feature_size][index:]
test_label = numpy_data[:, -1][index:]


train_dataset = MyDataset(train_data, train_label)
test_dataset = MyDataset(test_data, test_label)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = len(numpy_data) - batch_size, shuffle = False)



class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        Theta_x = self.logstic(x)
        out = F.softmax(self.logstic(x))
        return out, Theta_x


def loss_no_fair(output, target):
    loss = torch.mean((-target * torch.log(output)- (1 - target) * torch.log(1 - output)))

    return loss

def loss_with_fair(output, target, Theta_X, Sen, Sen_bar):
    pred_loss = torch.mean((-target * torch.log(output)- (1 - target) * torch.log(1 - output)))
    fair_loss = torch.mul(Sen - Sen_bar, Theta_X)
    #fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))
    fair_loss = torch.mean(fair_loss)

    return pred_loss

def agnostic_loss_no_fair(output, target, weight):

    pred_loss = (-target * torch.log(output)- (1 - target) * torch.log(1 - output))
    pred_loss = torch.mean(torch.mul(pred_loss, weight))
    return pred_loss


def robust_fair_loss(pred, label, Theta_X, weight, Sen, Sen_bar):
    # pred = torch.log(torch.from_numpy(np.array(pred,np.float)))
    # label = torch.log(torch.from_numpy(np.array(label, np.float)))
    # pred_loss = (-label * torch.log(pred)- (1 - label) * torch.log(1 - pred))
    # pred_loss = torch.mean(torch.mul(pred_loss, weight))


    Theta_X = torch.sum(Theta_X, dim=1)

    weight = weight.view(len(weight))
    Sen = Sen.view(len(Sen))
    Theta_X = Theta_X.view(len(Theta_X))
    fair_loss = torch.mul(torch.mul(weight, Sen) - Sen_bar, Theta_X)
    fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))

    return fair_loss


input_dim = feature_size - 1
output_dim = 2

model = logsticRegression(input_dim, output_dim)
use_gpu = torch.cuda.is_available()
# if use_gpu:
#     model = model.cuda()



criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def compute_model_parameter_fairness(weight, Sen, ratio, option, train_loader=train_loader, test_loader=test_loader):
    for epoch in range(num_epochs):
        print('*' * 10)
        print('epoch %d:' % (epoch+1))
        running_acc = 0.0
        model.train()
        for i,  (data, target) in enumerate(train_loader):
            img, label = data, target

            # weight = img[:, -1]
            img = img[:, :feature_size - 1]
            print(img.shape)


            label = label.data.numpy()
            for i in range(len(label)):
                label[i] = 0 if label[i] == 0.0 else 1
            label = torch.LongTensor(label)
            label = label.view(len(label))

            out, Theta_x = model(img)
            loss = criterion(out, label)
            loss_vector = loss


            loss = loss.view(len(loss), 1)
            weight = weight.view(len(weight), 1)
            loss = torch.mul(loss, weight)

            loss = loss.sum() / len(out)

            _, pred = torch.max(out.data, 1)

            print(loss)
            loss = loss + robust_fair_loss(pred, label, Theta_x, weight, Sen, ratio)
            print(loss)



            running_acc += torch.sum((pred==label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        running_acc = running_acc.numpy()
        # for p in model.parameters():
        #     print(p)
        print('Finish %d training epoch, Acc %.6f:' % (epoch, running_acc / len(train_label)))
        model.eval()
        theta = model.logstic.weight
        bias = model.logstic.bias

        running_acc = 0.0
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            label = label.data.numpy()
            for i in range(len(label)):
                label[i] = 0 if label[i] == 0.0 else 1
            label = torch.LongTensor(label)
            label = label.view(len(label))

            with torch.no_grad():
                out, _ = model(img)


            _, pred = torch.max(out.data, 1)


            running_acc += torch.sum((pred == label))
            Sen_test = numpy_data[:, 0][index:].reshape(-1, 1)

            Sen_test = torch.from_numpy(Sen_test).float()
            # count_S1 = torch.sum(Sen_test)
            # count_S0 = torch.sum(1 - Sen_test)
            pred = torch.where(pred > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            print(sum(pred))
            pred = pred.view(len(pred))
            Sen_test = Sen_test.view(len(Sen_test))
            count_Y1_S1 = torch.sum(torch.mul(pred, Sen_test))
            count_Y0_S1 = torch.sum(torch.mul(1.0 - pred, Sen_test))
            count_Y1_S0 = torch.sum(torch.mul(pred, 1.0 - Sen_test))
            count_Y0_S0 = torch.sum(torch.mul(1.0 - pred, 1.0 - Sen_test))


            # label = label.view(len(label), -1)

            r11 = count_Y1_S1 / len(Sen_test)
            r01 = count_Y0_S1 / len(Sen_test)
            r10 = count_Y1_S0 / len(Sen_test)
            r00 = count_Y0_S0 / len(Sen_test)
            risk_difference = abs(r11 / (r11 + r01) - r10 / (r10 + r00))

        running_acc = running_acc.numpy()
        print('Finish %d testing epoch, Acc %.6f:' % (epoch, running_acc / len(test_label)))
        print('Finish risk difference, risk difference %.6f' % (risk_difference))

    return theta.detach().numpy().transpose(), bias.detach().numpy(), loss_vector.detach().numpy(), Theta_x.detach().numpy()

Iteration = 20
sigma = 2
B = 5
Xtr = train_data
Ytr = train_label
ref_idx = np.random.randint(len(test_data), size = 500)
Xref = test_data[ref_idx, :]
# Sen = numpy_data[:, 0][0:index].reshape(-1, 1)
Sen_torch = torch.from_numpy(Sen).float()

weight = np.ones(len(train_data))
# upper = np.array(upper)
weight = torch.from_numpy(weight).float()
# weight = torch.from_numpy(upper).float()
ratio = 0.1
options = ["no_fair", "with_fair", "agnostic_fair", "agnostic_loss_no_fair"]
ratio_fair = np.sum(Sen) / len(Sen)
print(ratio_fair)
tau = 10
for i in range(Iteration):

    theta, bias, loss_vector, Theta_x = compute_model_parameter_fairness(weight, Sen_torch, ratio_fair, options[0])

    #weight, ratio = compute_weight_with_fairness_with_label(loss_vector, lower, upper, Theta_x, Sen, cluster)
    weight, ratio = compute_weight_with_fairness_no_label(loss_vector, lower, upper, Theta_x, Sen, cluster)
    # # # # # # # #print(weight[0:10])
    weight = torch.from_numpy(weight).float()







def train(X, Y, S, n_iters=20):

    batch_size = 35000
    learning_rate = 0.25
    num_epochs = 10

    train_data, train_label, lower, upper, S, cluster = create_data.weighted_adult_data_kmeans_wihtout_unlabel(numpy_data)


    weight = torch.from_numpy(np.ones(len(X))).float()
    ratio_fair = np.sum(S) / len(S)

    S_torch = torch.from_numpy(S).float()

    for i in range(n_iters):
        theta, _, loss_vector, Theta_x = compute_model_parameter_fairness(weight, S_torch, ratio_fair, None)
        weight, ratio = compute_weight_with_fairness_no_label(loss_vector, lower, upper, Theta_x, Sen, cluster)
        weight = torch.from_numpy(weight).float()