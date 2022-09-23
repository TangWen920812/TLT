import numpy as np
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import json
import torch
from scipy.ndimage import zoom
import torch.utils.data.dataloader
import importlib
import collections
from torch._six import string_classes, int_classes
from collections import OrderedDict
import copy
import functools
import torch.nn.functional as F

class TransLesionData(Dataset):
    def __init__(self, file_root, data_root, istrain=True):
        super(TransLesionData, self).__init__()
        with open(file_root, 'r') as file:
            self.data_dic = json.load(file)
        self.data_root = data_root
        self.istrain = istrain
        self.data_list = [k for k in self.data_dic if os.path.exists(os.path.join(data_root, 'img_fixed', k + '.nii'))]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        dic_name = self.data_list[item]
        shape_label = self.data_dic[dic_name]['target box'][-3:][::-1]
        moving_img_path = os.path.join(self.data_root, 'img_moving', dic_name + '.nii')
        moving_msk_path = os.path.join(self.data_root, 'msk_moving', dic_name + '.npy')
        fixed_img_path = os.path.join(self.data_root, 'img_fixed', dic_name + '.nii')
        fixed_msk_path = os.path.join(self.data_root, 'msk_fixed', dic_name + '.npy')
        deformed_moving_mask_path = os.path.join(self.data_root, 'deeds', dic_name + '_deformed_seg.nii.gz')
        moving_img = sitk.ReadImage(moving_img_path)
        moving_arr = sitk.GetArrayFromImage(moving_img).astype(float)
        fixed_img = sitk.ReadImage(fixed_img_path)
        fixed_arr = sitk.GetArrayFromImage(fixed_img).astype(float)
        moving_msk = np.load(moving_msk_path)
        fixed_msk = np.load(fixed_msk_path)
        deformed_msk = sitk.GetArrayFromImage(sitk.ReadImage(deformed_moving_mask_path))
        deformed_msk = (deformed_msk / 1000).astype(float)

        # print(moving_arr.shape)
        if moving_arr.shape[0] * moving_arr.shape[1] * moving_arr.shape[2] >= 40 * 200 * 200:
            # print('composition')
            hold_size_z, hold_size_y, hold_size_x = 40 // 2, 200 // 2, 200 // 2
            center = np.array(np.where(deformed_msk == deformed_msk.max()))[:, 0]
            center_z, center_y, center_x = center[0], center[1], center[2]
            shape = moving_arr.shape
            moving_arr = moving_arr[max(0, center_z - hold_size_z):min(shape[0], center_z + hold_size_z),
                                    max(0, center_y - hold_size_y):min(shape[1], center_y + hold_size_y),
                                    max(0, center_x - hold_size_x):min(shape[2], center_x + hold_size_x),]
            fixed_arr = fixed_arr[max(0, center_z - hold_size_z):min(shape[0], center_z + hold_size_z),
                                  max(0, center_y - hold_size_y):min(shape[1], center_y + hold_size_y),
                                  max(0, center_x - hold_size_x):min(shape[2], center_x + hold_size_x),]
            moving_msk = moving_msk[max(0, center_z - hold_size_z):min(shape[0], center_z + hold_size_z),
                                    max(0, center_y - hold_size_y):min(shape[1], center_y + hold_size_y),
                                    max(0, center_x - hold_size_x):min(shape[2], center_x + hold_size_x),]
            fixed_msk = fixed_msk[max(0, center_z - hold_size_z):min(shape[0], center_z + hold_size_z),
                                  max(0, center_y - hold_size_y):min(shape[1], center_y + hold_size_y),
                                  max(0, center_x - hold_size_x):min(shape[2], center_x + hold_size_x),]
            deformed_msk = deformed_msk[max(0, center_z - hold_size_z):min(shape[0], center_z + hold_size_z),
                                        max(0, center_y - hold_size_y):min(shape[1], center_y + hold_size_y),
                                        max(0, center_x - hold_size_x):min(shape[2], center_x + hold_size_x), ]
        center_label = np.array(np.where(fixed_msk==fixed_msk.max()))[:, 0]

        moving_arr = (moving_arr + 800) / 1600
        fixed_arr = (fixed_arr + 800) / 1600
        img_shape = np.array(fixed_arr.shape)
        cls_label = fixed_msk
        box_label = [center_label[0] / img_shape[0], center_label[1] / img_shape[1], center_label[2] / img_shape[2],
                     shape_label[0] / img_shape[0], shape_label[1] / img_shape[1], shape_label[2] / img_shape[2]]
        box_label = np.array(box_label)

        moving_arr = moving_arr[np.newaxis, ...]
        fixed_msk = deformed_msk[np.newaxis, ...]
        moving_msk = moving_msk[np.newaxis, ...]
        fixed_arr = fixed_arr[np.newaxis, ...]
        cls_label = cls_label[np.newaxis, ...]
        box_label = box_label[np.newaxis, ...]
        img_shape = img_shape[np.newaxis, ...]

        return moving_arr, moving_msk, fixed_arr, fixed_msk, cls_label, box_label, img_shape

class TensorDict(OrderedDict):
    """Container mainly used for dicts of torch tensors. Extends OrderedDict with pytorch functionality."""

    def concat(self, other):
        """Concatenates two dicts without copying internal data."""
        return TensorDict(self, **other)

    def copy(self):
        return TensorDict(super(TensorDict, self).copy())

    def __deepcopy__(self, memodict={}):
        return TensorDict(copy.deepcopy(list(self), memodict))

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorDict\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorDict({n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e for n, e in self.items()})
        return apply_attr

    def attribute(self, attr: str, *args):
        return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

    def apply(self, fn, *args, **kwargs):
        return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorDict, list))

class TensorList(list):
    """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

    def __init__(self, list_of_tensors = None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __deepcopy__(self, memodict={}):
        return TensorList(copy.deepcopy(list(self), memodict))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
        return TensorList([e + other for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
        return TensorList([other + e for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 - e2 for e1, e2 in zip(self, other)])
        return TensorList([e - other for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 - e1 for e1, e2 in zip(self, other)])
        return TensorList([other - e for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
        return TensorList([e * other for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
        return TensorList([other * e for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 / e2 for e1, e2 in zip(self, other)])
        return TensorList([e / other for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 / e1 for e1, e2 in zip(self, other)])
        return TensorList([other / e for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 @ e2 for e1, e2 in zip(self, other)])
        return TensorList([e @ other for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 @ e1 for e1, e2 in zip(self, other)])
        return TensorList([other @ e for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 % e2 for e1, e2 in zip(self, other)])
        return TensorList([e % other for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([e2 % e1 for e1, e2 in zip(self, other)])
        return TensorList([other % e for e in self])

    def __pos__(self):
        return TensorList([+e for e in self])

    def __neg__(self):
        return TensorList([-e for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 <= e2 for e1, e2 in zip(self, other)])
        return TensorList([e <= other for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([e1 >= e2 for e1, e2 in zip(self, other)])
        return TensorList([e >= other for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self

        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError('\'TensorList\' object has not attribute \'{}\''.format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))

def tensor_operation(op):
    def islist(a):
        return isinstance(a, TensorList)

    @functools.wraps(op)
    def oplist(*args, **kwargs):
        if len(args) == 0:
            raise ValueError('Must be at least one argument without keyword (i.e. operand).')

        if len(args) == 1:
            if islist(args[0]):
                return TensorList([op(a, **kwargs) for a in args[0]])
        else:
            # Multiple operands, assume max two
            if islist(args[0]) and islist(args[1]):
                return TensorList([op(a, b, *args[2:], **kwargs) for a, b in zip(*args[:2])])
            if islist(args[0]):
                return TensorList([op(a, *args[1:], **kwargs) for a in args[0]])
            if islist(args[1]):
                return TensorList([op(args[0], b, *args[2:], **kwargs) for b in args[1]])

        # None of the operands are lists
        return op(*args, **kwargs)

    return oplist

def _check_use_shared_memory():
    if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
        return getattr(torch.utils.data.dataloader, '_use_shared_memory')
    collate_lib = importlib.import_module('torch.utils.data._utils.collate')
    if hasattr(collate_lib, '_use_shared_memory'):
        return getattr(collate_lib, '_use_shared_memory')
    return torch.utils.data.get_worker_info() is not None

def ltr_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))

def ltr_collate_stack1(batch):
    """Puts each data field into a tensor. The tensors are stacked at dim=1 to form the batch"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _check_use_shared_memory():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 1, out=out)
        # if batch[0].dim() < 4:
        #     return torch.stack(batch, 0, out=out)
        # return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 1)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], TensorDict):
        return TensorDict({key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]})
    elif isinstance(batch[0], collections.Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], TensorList):
        transposed = zip(*batch)
        return TensorList([ltr_collate_stack1(samples) for samples in transposed])
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [ltr_collate_stack1(samples) for samples in transposed]
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))

class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Note: The only difference with default pytorch DataLoader is that an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.
    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        stack_dim (int): Dimension along which to stack to form the batch. (default: 0)
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)
    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use ``torch.initial_seed()`` to access the PyTorch seed for each
              worker in :attr:`worker_init_fn`, and use it to set other seeds
              before data loading.
    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate
            elif stack_dim == 1:
                collate_fn = ltr_collate_stack1
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        super(LTRLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim





