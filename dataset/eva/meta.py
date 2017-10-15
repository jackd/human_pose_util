cameras = ('C1', 'C2', 'C3', 'BW1', 'BW2', 'BW3', 'BW4')

subjects = ('S1', 'S2', 'S3', 'S4')
base_sequences = ('Combo', 'Gestures', 'Jog', 'ThrowCatch', 'Walking', 'Box')

test_subjects = 'S14',
test_base_sequences = 'Combo',

eval_subjects = ('S1', 'S2', 'S3')
eval_base_sequences = ('Gestures', 'Jog', 'ThrowCatch', 'Walking', 'Box')

# Trial 1 (index 0) validate/train partitions.
# Trial 2 is entirely test.
# Trial 3 is entirely train.
partitions = {
    "S1": {
        "Walking": (590, 1180),
        "Jog": (367, 735),
        "ThrowCatch": (473, 946),
        "Gestures": (395, 790),
        "Box": (385, 770)
    },
    "S2": {
        "Walking": (438, 877),
        "Jog": (398, 796),
        "ThrowCatch": (550, 1101),
        "Gestures": (500, 1000),
        "Box": (382, 765)
    },
    "S3": {
        "Walking": (448, 896),
        "Jog": (401, 803),
        "ThrowCatch": (493, 987),
        "Gestures": (533, 1067),
        "Box": (512, 1024)
    }
}

n_frames = {
    "S1": {
        "Walking": [1180, 980, 3238],
        "Jog": [735, 856, 3175],
        "ThrowCatch": [946, 929, 3453],
        "Gestures": [790, 1059, 2127],
        "Box": [770, 607, 1653],
        "Combo": [0, 2602, 0]
    },
    "S2": {
        "Walking": [877, 1097, 1523],
        "Jog": [796, 733, 1573],
        "ThrowCatch": [1101, 1346, 3340],
        "Gestures": [1000, 1025, 3551],
        "Box": [765, 975, 3108],
        "Combo": [0, 1996, 0]
    },
    "S3": {
        "Walking": [896, 806, 2358],
        "Jog": [803, 842, 1973],
        "ThrowCatch": [987, 967, 2074],
        "Gestures": [1067, 554, 1789],
        "Box": [1024, 719, 1573],
        "Combo": [0, 1761, 0]
    },
    "S4": {
        "Walking": [0, 670, 0],
        "Jog": [0, 593, 0],
        "ThrowCatch": [0, 776, 0],
        "Gestures": [0, 462, 0],
        "Box": [0, 557, 0],
        "Combo": [0, 1105, 0]
    },
}


def trial_index(sequence):
    return int(sequence.split('_')[1]) - 1


def sequence(base_sequence, index):
    return '%s_%d' % (base_sequence, index + 1)
