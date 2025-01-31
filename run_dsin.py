import numpy as np

from deepctr.models import DSIN
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_varlen_feature_names,get_fixlen_feature_names


def get_xy_fd(hash_flag=False):

    feature_columns = [SparseFeat('user', 3, hash_flag),
                       SparseFeat('gender', 2, hash_flag),
                       SparseFeat('item', 3 + 1, hash_flag),
                       SparseFeat('item_gender', 2 + 1, hash_flag),
                       DenseFeat('score', 1)]
    feature_columns += [VarLenSparseFeat('sess_0_item',3+1,4,use_hash=hash_flag,embedding_name='item'),VarLenSparseFeat('sess_0_item_gender',2+1,4,use_hash=hash_flag,embedding_name='item_gender')]
    feature_columns += [VarLenSparseFeat('sess_1_item', 3 + 1, 4, use_hash=hash_flag, embedding_name='item'),VarLenSparseFeat('sess_1_item_gender', 2 + 1, 4, use_hash=hash_flag,embedding_name='item_gender')]

    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    sess1_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [0, 0, 0, 0]])
    sess1_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [0, 0, 0, 0]])

    sess2_iid = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    sess2_igender = np.array([[1, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    sess_number = np.array([2, 1, 0])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'sess_0_item': sess1_iid, 'sess_0_item_gender': sess1_igender, 'score': score,
                    'sess_1_item': sess2_iid, 'sess_1_item_gender': sess2_igender, }

    fixlen_feature_names = get_fixlen_feature_names(feature_columns)
    varlen_feature_names = get_varlen_feature_names(feature_columns)
    x = [feature_dict[name] for name in fixlen_feature_names] + [feature_dict[name] for name in varlen_feature_names]

    x += [sess_number]

    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list

if __name__ == "__main__":
    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd(True)

    model = DSIN(feature_dim_dict, behavior_feature_list, sess_max_count=2, embedding_size=4,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, )

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)

