"""Dataset specification objects."""
import numpy as np
from human_pose_util.dataset.interface import Dataset
from human_pose_util.dataset.interface import MappedDataset, FilteredDataset
from human_pose_util.register import skeleton_register, dataset_register


class DatasetSpec(object):
    def __init__(self, attrs_spec, example_spec):
        self._example_spec = example_spec
        self._attrs_spec = attrs_spec

    def satisfied_by(self, dataset):
        return self._attrs_spec.satisfied_by(dataset.attrs) and \
            all([self._example_spec.satisfied_by(dataset[k]) for k in dataset])

    def assert_satisfied_by(self, dataset):
        self._attrs_spec.assert_satisfied_by(dataset.attrs)
        all([self._example_spec.assert_satisfied_by(
             dataset[k]) for k in dataset])


class AttrsSpec(object):
    def __init__(self, fn_dict):
        self._fn_dict = fn_dict

    def satisfied_by(self, attrs):
        return all(
            [k in attrs and v(attrs[k]) for k, v in self._fn_dict.items()])

    def assert_satisfied_by(self, attrs):
        for k, v in self._fn_dict.items():
            if k not in attrs:
                raise AssertionError('key %s not in attrs' % k)
            if not v(attrs[k]):
                raise AssertionError(
                    'validator failed for key-value pair (%s, %s)'
                    % (k, attrs[k]))


class ExampleSpec(object):
    def __init__(self, attrs_spec, data_specs):
        self._attrs_spec = attrs_spec
        self._data_specs = data_specs

    def satisfied_by(self, example):
        return self._attrs_spec.satisfied_by(example.attr) and \
            all([k in example for k in self._data_specs]) and \
            all([self._data_specs[k].satisfied_by(example[k])
                 for k in example])

    def assert_satisfied_by(self, example):
        self._attrs_spec.assert_satisfied_by(example.attrs)
        for k in self._data_specs:
            if k not in example:
                raise AssertionError('Key %k not in example' % k)
        for k in example:
            self._data_specs[k].assert_satisfied_by(example[k])


class DataSpec(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def satisfied_by(self, data):
        return data.dtype == self.dtype and all(
            [s is None or s == ds for s, ds in zip(self.shape, data.shape)])

    def assert_satisfied_by(self, data):
        if data.dtype != self.dtype:
            raise AssertionError(
                'Not correct datatype, %s vs %s' % (data.dtype, self.dtype))
        for i, (s, ds) in enumerate(zip(self.shape, data.shape)):
            if s is not None and s != ds:
                raise AssertionError('Shapes not consistent. %s vs %s'
                                     % (data.shape, self.shape))


def _is_str(x):
    return isinstance(x, str)


def _is_int(x):
    return isinstance(x, int)


def _is_len(n):
    def is_len(x):
        return len(x) == n
    return is_len


pose_sequence_example_attrs_spec = AttrsSpec({
    'subject_id': _is_str,
    'camera_id': _is_str,
    'sequence_id': _is_str,
    'f': _is_len(2),
    'c': _is_len(2),
    'r': _is_len(3),
    't': _is_len(3),
    'n_frames': _is_int,
    'fps': _is_int,
})

p3_spec = DataSpec((None, None, 3), np.float32)
p2_spec = DataSpec((None, None, 2), np.float32)

pose_sequence_example_data_specs = {
    'p3c': p3_spec,
    'p3w': p3_spec,
    'p2': p2_spec,
}

pose_sequence_example_spec = ExampleSpec(
    attrs_spec=pose_sequence_example_attrs_spec,
    data_specs=pose_sequence_example_data_specs)


def _is_num(x):
    return isinstance(x, (int, float))


pose_dataset_attrs_spec = AttrsSpec({
    'skeleton_id': lambda x: x in skeleton_register,
    'pixel_scale': _is_num
})

pose_dataset_spec = DatasetSpec(
    attrs_spec=pose_dataset_attrs_spec,
    example_spec=pose_sequence_example_spec)


def _is_float(x):
    return isinstance(x, float)


result_example_attr_spec = AttrsSpec(
    {'inference_time': _is_float, 'inference_fps': _is_float})

result_example_spec = ExampleSpec(
    attrs_spec=result_example_attr_spec,
    data_specs={
        'p3w': DataSpec((None, None), np.float32),
        'proc_error': DataSpec((None, None), np.float32),
        'sequence_proc_error': DataSpec((None, None), np.float32)
    }
)

result_dataset_attrs_spec = AttrsSpec({
    'model_id': _is_str,
    'dataset_id': lambda x: x in dataset_register,
})

result_dataset_spec = DatasetSpec(
    example_spec=result_example_spec,
    attrs_spec=result_dataset_attrs_spec)


def example_scaler(space_scale=1000, pixel_scale=1000):

    def scale_example(example):
        ss = space_scale(example) if callable(space_scale) else space_scale
        ps = pixel_scale(example) if callable(pixel_scale) else pixel_scale
        if ss == 1 and ps == 1:
            return example

        def scale_space(x):
            return x / ss

        def scale_pixels(x):
            return x / ps

        def inv_space_scale(x):
            return x * ss

        def inv_pixel_scale(x):
            return x * ps

        return MappedDataset(example, {
                'p3c': scale_space,
                'p3w': scale_space,
                'p2': scale_pixels,
            }, {
                'f': scale_pixels,
                'c': scale_pixels,
                't': scale_space,
                'pixel_scale': inv_pixel_scale,
                'space_scale': inv_space_scale
            })
    return scale_example


def scaled_dataset(base_dataset, pixel_scale=1, space_scale=1):
    if pixel_scale == 1 and space_scale == 1:
        return base_dataset
    scaler = example_scaler(pixel_scale=pixel_scale, space_scale=space_scale)
    return MappedDataset(base_dataset, scaler)


def height_normalized_dataset(base_dataset):
    heights = calculate_heights(base_dataset)

    def space_scale(example):
        return heights[example.attrs['subject_id']]

    return scaled_dataset(base_dataset, space_scale=space_scale)


def front_rotated_dataset(base_dataset, dim):
    from human_pose_util.transforms import np_impl
    skeleton = skeleton_register[base_dataset.attrs['skeleton_id']]

    def rotate_front(x):
        angle = skeleton.front_angle(x)
        if len(x.shape) == 3:
            angle = np.expand_dims(angle, axis=1)
        return np_impl.rotate_about(x, -angle, dim)

    return MappedDataset(base_dataset, {'p3w': rotate_front})


def origin_shifted_dataset(base_dataset):
    root_index = skeleton_register[
        base_dataset.attrs['skeleton_id']].root_index

    def shift(p3):
        p3 = p3.copy()
        p3[..., :2] -= p3[..., root_index:root_index+1, :2]
        return p3

    return MappedDataset(base_dataset, {'p3w': shift})


def normalized_dataset(
        dataset, height_normalized=True, front_rotated_dim=2,
        origin_shifted=True):
    if height_normalized:
        dataset = height_normalized_dataset(dataset)
    if front_rotated_dim is not None:
        dataset = front_rotated_dataset(dataset, front_rotated_dim)
    if origin_shifted:
        dataset = origin_shifted_dataset(dataset)
    return dataset


def filter_by_camera(base_dataset, camera_id):
    if not isinstance(camera_id, (list, tuple)):
        camera_id = [camera_id]
    return FilteredDataset(
        base_dataset, lambda ex: ex.attrs['camera_id'] in camera_id)


def filter_by_subjects(base_dataset, subject_ids):
    return FilteredDataset(
        base_dataset, lambda ex: ex.attrs['subject_id'] in subject_ids)


class ModifiedFpsExample(Dataset):
    def __init__(self, base_example, fps):
        self._base = base_example
        original_fps = self._base.attrs['fps']
        if original_fps % fps != 0:
            raise ValueError('original_fps must be divisble by fps.')
        take_every = original_fps // fps
        self._take_every = take_every
        self._attrs = {k: v for k, v in base_example.attrs.items()}

        self._attrs['fps'] = self._attrs['fps'] // take_every

    def __getitem__(self, key):
        if key in ['p3c', 'p3w', 'p2']:
            return self._base[key][::self._take_every]
        else:
            return self._base[key]

    @property
    def attrs(self):
        return self._attrs


def modified_fps_dataset(dataset, target_fps):
    def map_fn(example):
        return ModifiedFpsExample(example, target_fps)

    return MappedDataset(dataset, map_fn)


def calculate_heights(sequence_dataset):
    from human_pose_util.register import skeleton_register
    heights = {}
    skeleton = skeleton_register[sequence_dataset.attrs['skeleton_id']]

    def index(joints):
        if isinstance(joints, str):
            return skeleton.joint_index(joints)
        for j in joints:
            if skeleton.has_joint(j):
                return skeleton.joint_index(j)
        raise ValueError('No joint: %s' % str(joints))

    for key, example in sequence_dataset.items():
        p3 = example['p3w']
        subject_id = example.attrs['subject_id']
        height = skeleton.height(p3)
        if subject_id in heights:
            heights[subject_id] = max(heights[subject_id], height)
        else:
            heights[subject_id] = height
    return heights
