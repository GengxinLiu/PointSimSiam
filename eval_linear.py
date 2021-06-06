# evaluate the linear svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import torch
from data import ModelNet40EvalLinear, ModelNet10EvalLinear
from torch.utils.data import DataLoader
import numpy as np
from utils import accuracy
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def eval_encoder(encoder, data_name='ModelNet40', l2_norm=False, cls='svm', train_ratio=1., test_ratio=1.,
                 verbose=False, batch_size=20, seed=None):
    """ Evaluate the encoder.
    :param encoder: out put (bn, dim)
    :param data_name: ModelNet40 or ModelNet10
    :param l2_norm: makes the embeddings unit vectors.
    :param cls: classifier
    :param train_ratio: ratio of train data
    :param test_ratio: ratio of test data
    :param verbose:
    :param seed:
    :return: test accuracy
    """
    if seed is not None:
        np.random.seed(seed)
    encoder.eval()
    torch.set_grad_enabled(False)
    if cls == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif cls == 'svm':
        classifier = LinearSVC()
    else:
        raise ValueError('Invalid classifier')
    if data_name == 'ModelNet40':
        train_cls_loader = DataLoader(ModelNet40EvalLinear(ratio=train_ratio, train_cls=True),
                                      batch_size=batch_size, drop_last=False)
        test_cls_loader = DataLoader(ModelNet40EvalLinear(ratio=test_ratio, train_cls=False),
                                     batch_size=batch_size, drop_last=False)
    else:
        train_cls_loader = DataLoader(ModelNet10EvalLinear(ratio=train_ratio, train_cls=True),
                                      batch_size=batch_size, drop_last=False)
        test_cls_loader = DataLoader(ModelNet10EvalLinear(ratio=test_ratio, train_cls=False),
                                     batch_size=batch_size, drop_last=False)
    if verbose:
        print('get train data')
    train_x, train_y = [], []
    for batch in train_cls_loader:
        pts, labels = batch
        feats = encoder(pts.permute(0, 2, 1).cuda())
        if l2_norm:
            feats = F.normalize(feats, dim=1)
        train_x.extend(feats.cpu().numpy())
        train_y.extend(labels)
    train_x = np.array(train_x)
    if verbose:
        print(train_x.shape)
        print('train cls..')
    classifier.fit(train_x, train_y)
    train_acc = accuracy(classifier.predict(train_x), np.array(train_y))
    if verbose:
        print('train acc {:.3f}'.format(train_acc))
        print('get test data..')
    test_x, test_y = [], []
    for batch in test_cls_loader:
        pts, labels = batch
        feats = encoder(pts.permute(0, 2, 1).cuda()).cpu().numpy()
        test_x.extend(feats)
        test_y.extend(labels)
    test_x = np.array(test_x)
    test_acc = accuracy(classifier.predict(test_x), np.array(test_y))
    if verbose:
        print('test acc {:.3f}'.format(test_acc))
    return test_acc


def eval_epochs(log_tstamp, data_name='ModelNet40', l2_norm=False, cls='knn', train_ratio=1., test_ratio=1.,
                verbose=False, batch_size=20, seed=None):
    """ Evaluate the classification accuracy of epoch models.
    :param log_tstamp: file struct
                       --log_tstamp
                            --0.pth
                            --1.pth
                                ...
    :param l2_norm: makes the embeddings unit vectors.
    :param data_name: ModelNet40 or ModelNet10
    :param cls: classifier.
    :param train_ratio:
    :param test_ratio:
    :param verbose:
    :param seed:
    """
    files = sorted([int(f.split('.')[0]) for f in os.listdir(log_tstamp)])
    test_accs = []
    for f in files:
        path = os.path.join(log_tstamp, f'{f}.pth')
        print(f'eval model {path}..')
        acc = eval_encoder(torch.load(path), data_name, l2_norm, cls,
                           train_ratio, test_ratio, verbose, batch_size, seed)
        test_accs.append(acc)
        print('accuracy {:.3f}'.format(acc))
    plt.plot(test_accs)
    plt.tight_layout()
    plt.savefig(f'{log_tstamp}.png')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    encoder_path = 'log_simsiam/2021_05_10_06_50_08/epochs/60.pth'
    print('evaluate', encoder_path)
    # model = nn.DataParallel(torch.load(encoder_path))
    eval_encoder(torch.load(encoder_path), data_name='ModelNet40', l2_norm=False, cls='svm',
                 train_ratio=1.0, test_ratio=1.0, batch_size=20, verbose=True, seed=28)

    # epochs = list(range(45, 72))
    # logs = []
    # for e in epochs:
    #     encoder_path = f'log_simsiam/2021_04_05_19_46_41/epochs/{e}.pth'
    #     print('evaluate', encoder_path)
    #     model = nn.DataParallel(torch.load(encoder_path))
    #     acc = eval_encoder(model, data_name='ModelNet40', l2_norm=False, cls='svm',
    #                        train_ratio=1.0, test_ratio=1.0, batch_size=30, verbose=True, seed=28)
    #     logs.append('{}  {:.1f}'.format(encoder_path, acc * 100))
    #     print(logs[-1])
    # for l in logs:
    #     print(l)
