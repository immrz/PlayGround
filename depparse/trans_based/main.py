import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from depparse.trans_based.parser_utils import load_and_preprocess_data
from depparse.trans_based.parse import PartialParse
import argparse


class HyperParam:
    lr = 1e-3
    momentum = 0.9
    batch_size = 32
    num_epoch = 200
    hidden_size = 800


class TreeBank(Dataset):
    def __init__(self, example):
        self.ex = example
        self.num_ft = len(example[0][0])
        self.num_label = len(example[0][1])

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, item):
        features, legal_labels, gt = self.ex[item]
        features = torch.LongTensor(features)
        legal_labels = torch.DoubleTensor(legal_labels)
        gt = int(gt)
        return {'feature': features, 'legal_label': legal_labels, 'gt': gt}


class NaiveParser(nn.Module):
    def __init__(self, embed_mat, num_ft, num_label, hidden_size=800):
        super(NaiveParser, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings=torch.from_numpy(embed_mat), freeze=False)
        self.len_embed_ft = num_ft * embed_mat.shape[1]
        self.fc1 = nn.Linear(self.len_embed_ft + num_label, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_label)

    def forward(self, feature, legal_label):
        embed_ft = self.embed(feature)
        embed_ft = embed_ft.view(-1, self.len_embed_ft)
        if embed_ft.dtype != legal_label.dtype:
            legal_label = legal_label.type_as(embed_ft)
        in_ft = torch.cat([embed_ft, legal_label], dim=-1)
        hidden = self.fc1(in_ft)
        hidden = torch.tanh(hidden)
        return self.fc2(hidden)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('f', type=str, help='The folder datasets are located.')
    parser.add_argument('--model-path', type=str, help='Where to save and load the model.')
    parser.add_argument('--lr', type=float, help='The learning rate.')
    parser.add_argument('--momentum', type=float, help='The momentum value.')
    parser.add_argument('--batch-size', type=int, help='The size of each batch.')
    parser.add_argument('--num-epoch', type=int, help='The number of epochs.')
    parser.add_argument('--hidden-size', type=int, help='The size of the hidden layer')
    args = parser.parse_args()
    return args


def train_model(train_data, model, criterion, optimizer, scheduler, dev_data=None, batch_size=32, num_epoch=100):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_data, parser = dev_data

    for epoch in range(num_epoch):
        model.train()
        scheduler.step()
        epoch_loss = 0.0

        for sample in train_loader:
            feature, legal_label, gt = sample['feature'], sample['legal_label'], sample['gt']
            out = model(feature, legal_label)

            optimizer.zero_grad()
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print('Training epoch {:03d}, loss is {:.4f}'.format(epoch, epoch_loss))
        print('Evaluating on the development set...')
        prediction(model, eval_data, parser)
        print('\n' + '=' * 40 + '\n')
    return model


def prediction(model, data, parser):
    model.eval()
    with torch.set_grad_enabled(False):
        num_suc = 0
        num_correct = 0
        num_total = 0

        for id, example in enumerate(data):
            ex = PartialParse(example)
            ex.parse(model, parser)
            if not ex.success:
                continue
            num_suc += 1
            correct, n_words = ex.accuracy()
            num_correct += correct
            num_total += n_words

    print('Totally {:d} sentences, {:d} succeeded!'.format(len(data), num_suc))
    print('Accuracy: {:d} / {:d} = {:.4f}'.format(num_correct, num_total, num_correct / num_total))


def main():
    args = parse_args()

    parser, embeddings_matrix, train_examples, dev_set, test_set = load_and_preprocess_data(
        reduced=False, data_path=args.f)
    train_data = TreeBank(train_examples)

    default = HyperParam()
    if args.lr is not None:
        default.lr = args.lr
    if args.momentum is not None:
        default.momentum = args.momentum
    if args.batch_size is not None:
        default.batch_size = args.batch_size
    if args.num_epoch is not None:
        default.num_epoch = args.num_epoch
    if args.hidden_size is not None:
        default.hidden_size = args.hidden_size

    model = NaiveParser(embeddings_matrix, train_data.num_ft, train_data.num_label, hidden_size=default.hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=default.lr, momentum=default.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=default.num_epoch//3, gamma=0.1)

    model = train_model(train_data, model, criterion, optimizer, exp_lr_scheduler,
                        batch_size=default.batch_size, num_epoch=default.num_epoch)

    if args.model_path is not None:
        torch.save(model.state_dict(), args.model_path)

    prediction(model, test_set, parser)


if __name__ == '__main__':
    main()