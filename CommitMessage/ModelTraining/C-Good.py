import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


class ModelConfig:
    batch_size = 50
    output_size = 2
    hidden_dim = 384  # 768/2
    n_layers = 2
    lr = 1e-6
    bidirectional = False  # 这里为True，为Bi-LSTM
    drop_prob = 0.55
    # training params
    epochs = 10
    # batch_size=50
    print_every = 10
    clip = 5  # gradient clipping
    use_cuda = USE_CUDA
    bert_path = 'bert-base-uncased'  # 预训练bert路径
    save_path = 'bert_bilstm.pth'  # 模型保存路径
    sampleRate = 2


class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.bert = BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.bert(x)[0]
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        if self.bidirectional:
            hidden_last_L = hidden_last[-2]
            hidden_last_R = hidden_last[-1]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]
        out = self.dropout(hidden_last_out)
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2

        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, output, target):
        # convert output to pseudo probability
        out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
        probs = torch.sigmoid(out_target)
        focal_weight = torch.pow(1 - probs, self.gamma)

        # add focal weight to cross entropy
        ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            focal_loss = (focal_loss / focal_weight.sum()).sum()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss


def train_model(config, data_train):
    net = bert_lstm(config.bert_path,
                    config.hidden_dim,
                    config.output_size,
                    config.n_layers,
                    config.bidirectional,
                    config.drop_prob
                    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    if (config.use_cuda):
        net.cuda()
    net.train()
    for e in range(config.epochs):
        # initialize hidden state
        h = net.init_hidden(config.batch_size)
        counter = 0
        # batch loop
        for inputs, labels in data_train:
            counter += 1
            if (config.use_cuda):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output = net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            # loss stats
            if counter % config.print_every == 0:
                net.eval()
                with torch.no_grad():
                    val_h = net.init_hidden(config.batch_size)
                    val_losses = []

                net.train()
                print("Epoch: {}/{}, ".format(e + 1, config.epochs),
                      "Step: {}, ".format(counter),
                      "Loss: {:.6f}, ".format(loss.item()))
    torch.save(net.state_dict(), config.save_path)


def test_model(config, data_test):
    net = bert_lstm(config.bert_path,
                    config.hidden_dim,
                    config.output_size,
                    config.n_layers,
                    config.bidirectional,
                    config.drop_prob
                    )
    net.load_state_dict(torch.load(config.save_path))
    if (config.use_cuda):
        net.cuda()
    net.train()
    criterion = nn.CrossEntropyLoss()
    test_losses = []

    net.eval()

    classnum = 2
    res = []
    predict_num = torch.zeros((1, classnum))
    # init hidden state
    h = net.init_hidden(config.batch_size)
    net.eval()
    # iterate over test data
    for inputs, labels in data_test:
        h = tuple([each.data for each in h])
        if (USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs, h)

        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        _, pred = torch.max(output, 1)
        res.append(pred.cpu().numpy().tolist())
        pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
    print('predict_num', " ".join('%s' % id for id in predict_num))
    res = np.array(res).reshape(1, -1).tolist()
    return res[0]


def myDataProcess(dataFile):
    df = pd.read_csv("~/" + str(dataFile), encoding='UTF-8')

    labeledDF = df[df.label.notnull() & df.if_mulit_commit.isnull()]
    labeledDF["new_message1"].apply(lambda x: x.replace('<enter>', '$enter').replace('<tab>', '$tab'). \
                                    replace('<url>', '$url').replace('<version>', '$version') \
                                    .replace('<pr_link>', '$pull request>').replace('<issue_link >',
                                                                                    '$issue') \
                                    .replace('<otherCommit_link>', '$other commit').replace("<method_name>",
                                                                                            "$method") \
                                    .replace("<file_name>", "$file").replace("<iden>", "$token"))
    # 包含what和why的是正样本，标记为1
    whyLabels = labeledDF['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (1 if x == 2.0 else (0 if x == 3.0 else 1))))
    whatLabels = labeledDF['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (1 if x == 3.0 else 0))))
    Labels = labeledDF['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (0 if x == 3.0 else 1))))
    print("load data successfully!")
    ifAllTextFile = labeledDF["if_all_text_file"]
    ifAllCodeFile = labeledDF["if_all_code_file"]
    changeLines = labeledDF["change_lines"]
    developerExpertise = labeledDF["developer_expertise"]
    commitDatePos = labeledDF["commit_date_position"]
    messages = list(labeledDF['new_message1'].array)

    return messages, np.array(whyLabels), np.array(whatLabels), np.array(Labels), np.array(ifAllTextFile), np.array(
        ifAllCodeFile), np.array(changeLines), np.array(developerExpertise), np.array(commitDatePos)


if __name__ == '__main__':
    os.system("/home/tyc/srun-linux")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_config = ModelConfig()
    text, whyLabels, whatLabels, Labels, ifAllTextFile, ifAllCodeFile, changeLines, developerExpertise, commitDatePos = myDataProcess(
        "message_sample.csv")
    result_comments = text
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)

    result_comments_id = tokenizer(result_comments,
                                   padding=True,
                                   truncation=True,
                                   max_length=200,
                                   return_tensors='pt')
    X = result_comments_id['input_ids']
    y = torch.from_numpy(whyLabels).float()
    yWhat = torch.from_numpy(whatLabels).float()
    yAll = torch.from_numpy(Labels).float()

    fold = KFold(n_splits=10, random_state=6666, shuffle=True)

    PRECISION = np.array([0.0, 0.0])
    RECALl = np.array([0.0, 0.0])
    F1 = np.array([0.0, 0.0])
    ACC = 0.0
    fold_num = 1
    WeightedPRECISION = 0.0
    WeightedRECALl = 0.0
    WeightedF1 = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for train_index, test_index in fold.split(X, y):
        X_train, X_test, y_train, y_test = \
            X[train_index], X[test_index], y[train_index], y[test_index]

        print("train_label: %s" % str(sorted(Counter(y_train).items())))
        yWhat_train, yWhat_test = \
            yWhat[train_index], yWhat[test_index]

        yAll_train, yAll_test = \
            yAll[train_index], yAll[test_index]

        # why training
        posNum = np.sum(whyLabels == 1)
        negNum = (int)(posNum / model_config.sampleRate)
        XWhy_train, y_train = RandomOverSampler(sampling_strategy={1: posNum, 0: negNum},
                                                random_state=666).fit_resample(
            X_train, y_train)
        print("train_label: %s" % str(sorted(Counter(y_train).items())))
        # XWhy_train = X_train
        XWhy_train = torch.from_numpy(XWhy_train)
        y_train = torch.from_numpy(y_train)
        train_data = TensorDataset(XWhy_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        print(len(XWhy_train))
        print(len(y_train))
        print(len(X_test))

        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=model_config.batch_size,
                                  drop_last=True)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=model_config.batch_size,
                                 drop_last=True)
        if (USE_CUDA):
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')
        train_model(model_config, train_loader)

        predWhy = test_model(model_config, test_loader)

        # what training
        posNum = np.sum(whatLabels == 1)
        negNum = (int)(posNum / 1.5)
        XWhat_train, yWhat_train = RandomOverSampler(sampling_strategy={1: posNum, 0: negNum},
                                                     random_state=666).fit_resample(
            X_train, yWhat_train)
        print("train_label: %s" % str(sorted(Counter(yWhat_train).items())))
        # XWhat_train = X_train
        XWhat_train = torch.from_numpy(XWhat_train)
        yWhat_train = torch.from_numpy(yWhat_train)
        train_data = TensorDataset(XWhat_train, yWhat_train)
        test_data = TensorDataset(X_test, yWhat_test)
        print(len(XWhat_train))
        print(len(yWhat_train))
        print(len(X_test))

        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=model_config.batch_size,
                                  drop_last=True)
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=model_config.batch_size,
                                 drop_last=True)
        if (USE_CUDA):
            print('Run on GPU.')
        else:
            print('No GPU available, run on CPU.')
        train_model(model_config, train_loader)

        predWhat = test_model(model_config, test_loader)
        print(len(predWhat))
        print(len(predWhy))
        print(len(yAll_test))
        for i in range(0, len(predWhy)):
            why, what, tar = predWhy[i], predWhat[i], yAll_test[i]
            if (why + what) == 2 and tar == 1:
                tp += 1
            elif (why + what) == 2 and tar == 0:
                fp += 1
            elif (why + what) != 2 and tar == 1:
                fn += 1
            elif (why + what) != 2 and tar == 0:
                tn += 1

        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        NegativePrecision = 0 if tn + fn == 0 else tn / (tn + fn)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        NegativeRecall = 0 if tn + fp == 0 else tn / (tn + fp)
        F1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        NegativeF1 = 0 if (NegativePrecision + NegativeRecall) == 0 else (2 * NegativeRecall * NegativePrecision) / (
                NegativePrecision + NegativeRecall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(tp, fp, tn, fn)
        print('Total Accuracy', accuracy)
        print('Total Precision', [NegativePrecision, precision])
        print('Total Recall', [NegativeRecall, recall])
        print('Total F1', [NegativeF1, F1])
