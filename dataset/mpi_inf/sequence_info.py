_data_labels = [
    'bg_augmentable',
    'ub_augmentable',
    'lb_augmentable',
    'chair_augmentable',
    'fps',
    'num_frames'
]

_sequence_info = [
    # S1
    [
        [True, False, False, True, 25, 6416],  # Seq 1
        [False, True, False, True, 50, 12430],  # Seq 2
    ],
    # S2
    [
        [True, False, False, True, 25, 6502],  # Seq 1
        [True, True, True, True, 25, 6081],  # Seq 2
    ],
    # S3
    [
        [True, False, False, True, 50, 12488],  # Seq 1
        [True, True, True, True, 50, 12283],  # Seq 2
    ],
    # S4
    [
        [True, False, False, True, 25, 6171],  # Seq 1
        [False, True, False, True, 25, 6675],  # Seq 2
    ],
    # S5
    [
        [True, False, False, True, 50, 12820],  # Seq 1
        [True, True, True, True, 50, 12312],  # Seq 2
    ],
    # S6
    [
        [True, False, False, True, 25, 6188],  # Seq 1
        [True, True, True, True, 25, 6145],  # Seq 2
    ],
    # S7
    [
        [True, True, True, True, 25, 6239],  # Seq 1
        [True, False, False, True, 25, 6320],  # Seq 2
    ],
    # S8
    [
        [True, True, True, True, 25, 6468],  # Seq 1
        [True, False, False, True, 25, 6054],  # Seq 2
    ],
]


def get_sequence_info(subject_idx, sequence_idx):
    data = _sequence_info[subject_idx][sequence_idx]
    return {k: v for k, v in zip(_data_labels, data)}


if __name__ == '__main__':
    for k, v in get_sequence_info(0, 0).items():
        print('%s: %s' % (k, v))
