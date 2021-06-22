import torch
from torch.utils import data
import numpy as np
import os
import pickle as pkl

idx_lookup = {
    'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
    'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
    'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
    'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
    'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
    'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
    'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
    'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39
}

SHAPENET_PATH = "../ShapeNetPoints_2048_pkl"
SHAPENET_PARTIALSCAN_1_PATH = "../ShapeNetScan"
SHAPENET_PARTIALSCAN_2_PATH = "../ShapeNetScan_2"
MODELNET40_PATH = "../ModelNet40"


def load_shapenet(category, file_id):
    """
    :param category: point cloud category
    :param file_id: file id
    """
    f = open(f'{SHAPENET_PATH}/{category}/{file_id}.pkl', 'rb')
    return pkl.load(f)


def load_shapenet_partial_1(category, file_id, view_id):
    """
    :param category: point cloud category.
    :param file_id: file id.
    :param view_id: partial scan view id.
    """
    f = open(f'{SHAPENET_PARTIALSCAN_1_PATH}/{category}/{file_id}/{view_id}', 'rb')
    return pkl.load(f)


def load_shapenet_partial_2(category, file_id, view_id):
    """
    :param category: point cloud category.
    :param file_id: file id.
    :param view_id: partial scan view id.
    """
    f = open(f'{SHAPENET_PARTIALSCAN_2_PATH}/{category}/{file_id}/{view_id}', 'rb')
    return pkl.load(f)


class ShapeNetUnsup(data.Dataset):
    """
    ShapeNetCore Dataset for unsupervised learning.
    """

    def __init__(self, num_points=2048, txt_file='path_to_train_txt'):
        """
        Dataset for DataLoader.
        """
        super(ShapeNetUnsup).__init__()
        f = open(txt_file, 'r')
        self.num_points = num_points
        self.files = f.readlines()

    def __getitem__(self, index):
        """
        :param index:
        :return: pts (n, 3).
        """
        category, file_id = self.files[index].split()
        # load original data
        points = load_shapenet(category, file_id)
        inds = np.random.choice(len(points), size=self.num_points, replace=False)
        points = points[inds]
        return points

    def __len__(self):
        return len(self.files)


class ShapeNetScanUnsup(data.Dataset):
    """
    ShapeNetScan Dataset for unsupervised learning.
    """

    def __init__(self, num_points=1024, txt_file='path_to_train_txt', data_id=1):
        """
        Dataset for DataLoader.
        :param num_points: sample points.
        :param data_id: id of shapnetscan, 1 or 2
        """
        super(ShapeNetScanUnsup).__init__()
        self.num_points = num_points

        f = open(txt_file, 'r')
        self.files = f.readlines()
        if data_id == 1:
            self.SHAPENET_PARTIALSCAN_PATH = SHAPENET_PARTIALSCAN_1_PATH
            self.load_fun = load_shapenet_partial_1
        else:
            self.SHAPENET_PARTIALSCAN_PATH = SHAPENET_PARTIALSCAN_2_PATH
            self.load_fun = load_shapenet_partial_2

    def __getitem__(self, index):
        """
        :param index:
        :return: original points, partial scan points, (num_points, 3)
        """
        category, file_id = self.files[index].split()
        # load original data
        points = load_shapenet(category, file_id)
        inds = np.random.choice(len(points), size=self.num_points, replace=False)
        points = points[inds]

        # load partial scan data
        view_id = np.random.choice(os.listdir(os.path.join(self.SHAPENET_PARTIALSCAN_PATH, category, file_id)), size=1,
                                   replace=False)[0]  # sample one views.
        partial_data = self.load_fun(category, file_id, view_id)
        inds = np.random.choice(len(partial_data), size=self.num_points, replace=False)
        partial_data = partial_data[inds]

        return points, partial_data

    def __len__(self):
        return len(self.files)


class RandomTransform(object):
    def __init__(
            self, scale=True, translate=True, jitter=True, rotate_z=True, rotate_group=True,
            scale_low=0.5, scale_high=1.5, shift_range=0.2, sigma=0.01, rot_range=30
    ):
        self.ops = []
        if scale:
            self.ops.append(self.random_scale(scale_low, scale_high))
        if translate:
            self.ops.append(self.random_translate(shift_range))
        if jitter:
            self.ops.append(self.random_jitter(sigma))
        if rotate_z:
            self.ops.append(self.random_rotate_z())
        if rotate_group:
            self.ops.append(self.random_rotate_group(rot_range))

    def __call__(self, batch_data):
        if len(self.ops) == 0:
            return batch_data
        if type(batch_data) == torch.Tensor:
            batch_data = batch_data.numpy()
        for ops in self.ops:
            batch_data = ops(batch_data)
        return batch_data

    @staticmethod
    def random_scale(scale_low=0.5, scale_high=1.5):
        """ Randomly scale the point cloud.
        :param scale_low: lower bound of scale
        :param scale_high: higher bound of scale
        """

        def _scale(batch_data: np.ndarray):
            """
            :param batch_data: original batch of point clouds, (bn, n, 3)
            :return:  (bn, n, 3) scaled batch of point clouds
            """
            B, N, C = batch_data.shape
            scales = np.random.uniform(scale_low, scale_high, B)
            for batch_index in range(B):
                batch_data[batch_index, :, :] *= scales[batch_index]
            return batch_data

        return _scale

    @staticmethod
    def random_translate(shift_range=0.2):
        """ Randomly shift point cloud. Shift is per point cloud.
        :param shift_range:
        """

        def _translate(batch_data: np.ndarray):
            """
            :param batch_data: original batch of point clouds, (bn, n, 3)
            :return: (bn, n, 3) random translate batch of point clouds
            """
            B, N, C = batch_data.shape
            shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
            for batch_index in range(B):
                batch_data[batch_index, :, :] += shifts[batch_index, :]
            return batch_data

        return _translate

    @staticmethod
    def random_jitter(sigma=0.01):
        """ Randomly jitter points
        :param sigma:
        """

        def _jitter(batch_data: np.ndarray):
            """
            :param batch_data: original batch of point clouds, (bn, n, 3)
            :return: (bn, n, 3) random jittered batch of point clouds
            """
            B, N, C = batch_data.shape
            jittered_data = sigma * np.random.randn(B, N, C)
            jittered_data += batch_data
            return jittered_data

        return _jitter

    @staticmethod
    def random_rotate_z():
        """ Randomly rotate the point clouds along z aixs
        """

        def _rotate_z(batch_data: np.ndarray):
            """
            :param batch_data:
            :return: (bn, n, 3) random rotate batch of point clouds
            """
            rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
            for k in range(batch_data.shape[0]):
                rotation_angle = np.random.uniform() * 2 * np.pi
                cosval = np.cos(rotation_angle)
                sinval = np.sin(rotation_angle)
                rotation_matrix = np.array([[cosval, sinval, 0],
                                            [-sinval, cosval, 0],
                                            [0, 0, 1]])
                shape_pc = batch_data[k, ...]
                rotated_data[k, ...] = np.dot(
                    shape_pc.reshape((-1, 3)), rotation_matrix)
            return rotated_data

        return _rotate_z

    @staticmethod
    def random_rotate_group(rot_range=60):
        """ Randomly rotate the point clouds
        """

        def _random_rotate_group(batch_data: np.ndarray):
            """
            :param batch_data:
            :return: (bn, n, 3) random rotate batch of point clouds
            """
            affines = []
            min_rot, max_rot = -rot_range, rot_range
            for i in range(len(batch_data)):
                a = np.radians(np.random.rand() *
                               (max_rot - min_rot) + min_rot)
                AX = np.array([[1, 0, 0],
                               [0, np.cos(a), -
                               np.sin(a)],
                               [0, np.sin(a), np.cos(a)]], dtype=np.float32)

                a = np.radians(np.random.rand() *
                               (max_rot - min_rot) + min_rot)
                AY = np.array([[np.cos(a), 0, np.sin(a)],
                               [0, 1, 0],
                               [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)

                a = np.radians(np.random.rand() *
                               (max_rot - min_rot) + min_rot)
                AZ = np.array([[np.cos(a), - np.sin(a), 0.],
                               [np.sin(a), np.cos(a), 0.],
                               [0., 0., 1.]], dtype=np.float32)
                A = np.matmul(np.matmul(AX, AY), AZ)
                affines.append(A)
            return np.matmul(batch_data, np.array(affines))

        return _random_rotate_group


class RandomCrop(object):
    def __init__(self, crop_size=None, crop_norm=False):
        """ Random crop a point cloud.
        :param crop_size: size of crop patch.
        :param crop_norm: normalize the crop.
        """
        self.crop_size = crop_size
        self.crop_norm = crop_norm

    def __call__(self, batch_data: np.ndarray):
        if self.crop_size is None:
            return batch_data
        B, N, C = batch_data.shape
        # random select anchor points.
        batch_anchor_idxs = np.random.randint(0, N, size=B, dtype=int)
        batch_anchor = []
        for i in range(B):
            batch_anchor.append(batch_data[i][batch_anchor_idxs[i]])
        batch_anchor = np.expand_dims(batch_anchor, axis=1)
        dist = -np.sum((batch_anchor - batch_data) ** 2, axis=-1)  # (B, N)
        crop_size = np.random.choice(self.crop_size, size=1)[0]
        nearest_val, nearest_inds = torch.topk(
            torch.from_numpy(dist), k=crop_size, dim=1)
        nearest_inds = nearest_inds.numpy()
        crop_points = []
        for i in range(B):
            crop = batch_data[i][nearest_inds[i]]
            if self.crop_norm:
                crop -= np.mean(crop, axis=0)
            crop_points.append(crop)
        return np.array(crop_points)


class RandomPart(object):
    def __init__(self, num_part=200, return_label=True):
        self.return_label = return_label
        self.num_part = num_part

    def get_part(self, shape: np.ndarray):
        """
        :param shape: (n, 3)
        :return: (m, 3), label
        """
        # numpy load
        x, y, z = shape[:, 0], shape[:, 1], shape[:, 2]
        # x>0 y>0 z>0
        pos_x, pos_y, pos_z = set(np.where(x > 0)[0]), set(
            np.where(y > 0)[0]), set(np.where(z > 0)[0])
        # x<0 y<0 z<0
        neg_x, neg_y, neg_z = set(np.where(x <= 0)[0]), set(
            np.where(y <= 0)[0]), set(np.where(z <= 0)[0])
        conditions = [
            [pos_x, pos_y, pos_z], [pos_x, pos_y, neg_z], [
                pos_x, neg_y, pos_z], [neg_x, pos_y, pos_z],
            [pos_x, neg_y, neg_z], [neg_x, pos_y, neg_z], [
                neg_x, neg_y, pos_z], [neg_x, neg_y, neg_z]
        ]
        while True:
            label = np.random.choice(len(conditions), size=1)
            cx, cy, cz = conditions[int(label)]
            part_inds = np.array(
                list(cx.intersection(cy).intersection(cz)), dtype=int)
            part_points = shape[part_inds]
            if len(part_inds) < 100:
                continue
            # sample fix size part
            replace = True if len(part_points) < self.num_part else False
            sample_ind = np.random.choice(
                part_inds, size=self.num_part, replace=replace)
            part_points = shape[np.array(sample_ind, dtype=int)]
            part_points -= np.mean(part_points, axis=0)  # move to center
            break
        return part_points, label

    def __call__(self, batch_data: np.ndarray):
        """
        :param batch_data: (b, n, 3)
        :return part: (m, 3) part of the shape.
        :return label: label of the part.
        """
        parts, labels = [], []
        for point in batch_data:
            part, label = self.get_part(point)
            parts.append(part)
            labels.append(label)
        if self.return_label:
            return np.array(parts), np.array(labels, dtype=int)
        else:
            return np.array(parts)


class ModelNet40Unsup(data.Dataset):
    """
    ModelNet40 Dataset for unsupervised learning.
    """

    def __init__(self, data_root=MODELNET40_PATH, add_test=False, train_ratio=1.):
        """
        Dataset for DataLoader.
        :param data_root: data path.
        :param add_test: add test dataset.
        :param train_ratio: use data ratio.
        """
        super(ModelNet40Unsup).__init__()
        self.files = []
        for c in os.listdir(data_root):
            # c: airplane, bed...
            c_path_train = os.path.join(data_root, c, 'train')
            sample_files = np.random.choice(
                os.listdir(c_path_train),
                size=int(train_ratio * len(os.listdir(c_path_train))),
                replace=False
            )
            for file in sample_files:
                self.files.append(os.path.join(c_path_train, file))
            if add_test:
                c_path_test = os.path.join(data_root, c, 'test')
                sample_files = np.random.choice(
                    os.listdir(c_path_test),
                    size=int(train_ratio * len(os.listdir(c_path_test))),
                    replace=False
                )
                for file in sample_files:
                    self.files.append(os.path.join(c_path_test, file))

    def __getitem__(self, index):
        """
        :param index:
        :return: pts (n, 3).
        """
        return np.load(self.files[index])

    def __len__(self):
        return len(self.files)


class ModelNet40EvalLinear(data.Dataset):
    """
    ModelNet40 Dataset for linear svm.
    """

    def __init__(self, data_root=MODELNET40_PATH, ratio=1., train_cls=True):
        """
        Dataset for DataLoader.
        :param data_root: data path.
        :param ratio: use data ratio.
        :param train_cls: bool, train classifier(return train data) or eval classifier(return test data).
        """
        super(ModelNet40EvalLinear).__init__()
        self.files = []
        self.labels = []

        dir = 'train' if train_cls else 'test'
        for c in os.listdir(data_root):
            # c: airplane, bed...
            c_path = os.path.join(data_root, c, dir)
            self.labels.extend([idx_lookup[c]] * len(os.listdir(c_path)))
            for file in os.listdir(c_path):
                self.files.append(os.path.join(c_path, file))

        num = len(self.files)
        sample = np.random.choice(num, size=int(ratio * num), replace=False)
        self.files = np.array(self.files)[sample]
        self.labels = np.array(self.labels)[sample]

    def __getitem__(self, index):
        """
        :param index:
        :return: pts (n, 3), label.
        """
        path = self.files[index]
        pts = torch.from_numpy(np.load(path)).float()
        return pts, self.labels[index]

    def __len__(self):
        return len(self.files)


class ModelNet10EvalLinear(data.Dataset):
    """
    ModelNet10 Dataset for linear svm.
    """

    def __init__(self, data_root=MODELNET40_PATH, ratio=1., train_cls=True):
        """
        Dataset for DataLoader.
        :param data_root: data path.
        :param ratio: use data ratio.
        :param train_cls: bool, train classifier(return train data) or eval classifier(return test data).
        """
        super(ModelNet40EvalLinear).__init__()
        self.files = []
        self.labels = []
        cats = [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'
        ]  # ModelNet10 class
        dir = 'train' if train_cls else 'test'
        for c in cats:
            # c: airplane, bed...
            c_path = os.path.join(data_root, c, dir)
            self.labels.extend([idx_lookup[c]] * len(os.listdir(c_path)))
            for file in os.listdir(c_path):
                self.files.append(os.path.join(c_path, file))

        num = len(self.files)
        sample = np.random.choice(num, size=int(ratio * num), replace=False)
        self.files = np.array(self.files)[sample]
        self.labels = np.array(self.labels)[sample]

    def __getitem__(self, index):
        """
        :param index:
        :return: pts (n, 3), label.
        """
        path = self.files[index]
        # pts = torch.from_numpy(np.load(path)).float()
        pts = np.load(path)
        return pts, self.labels[index]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    data = ModelNet10EvalLinear(train_cls=False)
