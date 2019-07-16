import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences

from Recommender import Recommender
from utils import *
import numpy as np


# A PyTorch implementation of Convolutional Sequence Embedding Recommendation Model (Caser)
# https://github.com/graytowne/caser_pytorch

class CaserModel(Recommender):
    def __init__(self, uNum, iNum, dim, maxlen, use_cuda=False):

        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.maxlen = maxlen
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self._net = Caser(self.uNum,
                          self.iNum,
                          self.dim, self.maxlen).to(self._device)

        self._optimizer = optim.Adam(self._net.parameters())

    def init(self, df):
        self.df = df

    def get_train_instances(self, train):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """
        sequence_length = self.maxlen
        target_length = 3  # default paper is 3, for a fair comparison we use 1 for all baselines

        users, checkins, positive_venues, negative_venues = [], [], [], []

        for u in range(self.uNum):
            visited = self.df[self.df.uid == u].iid.tolist()
            if len(visited) < self.maxlen + 1:
                continue

            for i in range(len(visited) - sequence_length):
                users.append([u])
                checkins.append(visited[i:i + sequence_length])
                positive_venues.append(visited[i + sequence_length: i + sequence_length + target_length])

                neg = []
                for j in range(target_length):

                    n = np.random.randint(0, self.iNum)
                    while (u, n) in train:
                        n = np.random.randint(0, self.iNum)
                    neg.append(n)
                negative_venues.append(neg)

        return [np.array(users), np.array(checkins), np.array(positive_venues), np.array(negative_venues)], None

    def load_pre_train(self, pre):
        super().load_pre_train(pre)

    def save(self, path):
        super().save(path)

    def get_params(self):
        return ""

    def minibatch(self, *tensors, **kwargs):

        batch_size = kwargs.get('batch_size', 128)

        if len(tensors) == 1:
            tensor = tensors[0]
            for i in range(0, len(tensor), batch_size):
                yield tensor[i:i + batch_size]
        else:
            for i in range(0, len(tensors[0]), batch_size):
                yield tuple(x[i:i + batch_size] for x in tensors)

    def train(self, x_train, y_train, batch_size):

        # set model to training mode
        self._net.train()

        users_np, sequences_np, targets_np, negatives_np = x_train

        # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
        users, sequences, targets, negatives = (torch.from_numpy(users_np).long(),
                                                torch.from_numpy(sequences_np).long(),
                                                torch.from_numpy(targets_np).long(),
                                                torch.from_numpy(negatives_np).long())

        users, sequences, targets, negatives = (users.to(self._device),
                                                sequences.to(self._device),
                                                targets.to(self._device),
                                                negatives.to(self._device))

        losses = []
        for (minibatch_num,
             (batch_users,
              batch_sequences,
              batch_targets,
              batch_negatives)) in enumerate(self.minibatch(users,
                                                            sequences,
                                                            targets,
                                                            negatives,
                                                            batch_size=batch_size)):
            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            items_prediction = self._net(batch_sequences,
                                         batch_users,
                                         items_to_predict)

            (targets_prediction,
             negatives_prediction) = torch.split(items_prediction,
                                                 [batch_targets.size(1),
                                                  batch_negatives.size(1)], dim=1)

            self._optimizer.zero_grad()
            # compute the binary cross-entropy loss
            positive_loss = -torch.mean(
                torch.log(torch.sigmoid(targets_prediction)))
            negative_loss = -torch.mean(
                torch.log(1 - torch.sigmoid(negatives_prediction)))
            loss = positive_loss + negative_loss

            losses.append(loss.item())

            loss.backward()
            self._optimizer.step()

        return "%.4f" % np.mean(losses)

    def rank(self, users, items):
        """
        Make predictions for evaluation: given a user id, it will
        first retrieve the test sequence associated with that user
        and compute the recommendation scores for items.

        Parameters
        ----------

        user_id: int
           users id for which prediction scores needed.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        """

        # set model to evaluation model
        self._net.eval()
        with torch.no_grad():
            sequences_np = pad_sequences([self.df[self.df.uid == users[0]].iid.values], self.maxlen)
            sequences_np = np.atleast_2d(sequences_np)
            sequences = torch.from_numpy(sequences_np).long()
            item_ids = torch.from_numpy(items).long()
            user_id = torch.from_numpy(np.array([[users[0]]])).long()

            user, sequences, items = (user_id.to(self._device),
                                      sequences.to(self._device),
                                      item_ids.to(self._device))

            out = self._net(sequences,
                            user,
                            items,
                            for_pred=True)

        return out.cpu().numpy().flatten()


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, dim, maxlen):
        super(Caser, self).__init__()
        # init args
        L = maxlen
        dims = dim
        self.n_h = 16
        self.n_v = 4
        self.drop_ratio = 0.5
        self.ac_conv = activation_getter['relu']
        self.ac_fc = activation_getter['relu']

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims + dims)
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res



