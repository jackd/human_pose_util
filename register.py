class Register(object):
    """Thin wrapper around a dictionary that ensures no overwriting."""
    def __init__(self, validator_fn=None):
        self._registry = {}
        self._validator_fn = validator_fn

    def __setitem__(self, key, value):
        if key in self._registry:
            if self._registry[key] != value:
                raise KeyError(
                    'Different value already registered for %s' % key)
        else:
            if self._validator_fn is None or self._validator_fn(value):
                self._registry[key] = value
            else:
                raise ValueError('Failed validation: %s' % key)

    def __getitem__(self, key):
        if key not in self._registry:
            raise KeyError('No value registered for key %s' % key)
        return self._registry[key]

    def __delete__(self, key):
        del self._registry[key]

    def __contains__(self, key):
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()


dataset_register = Register()
skeleton_register = Register()
registers = Register()
registers['skeleton'] = skeleton_register
registers['dataset'] = dataset_register


def register_default_datasets():
    from dataset.h3m.dataset import register_h3m_defaults
    from dataset.eva.dataset import register_eva_defaults
    register_h3m_defaults()
    register_eva_defaults()


def register_default_skeletons():
    from dataset.h3m.skeleton import s24
    from dataset.eva.skeleton import s14, s16, s20
    # h3m skeleton
    skeleton_register['s24'] = s24
    # eva skeletons
    for k, v in [['s20', s20], ['s16', s16], ['s14', s14]]:
        skeleton_register[k] = v


def register_defaults():
    register_default_skeletons()
    register_default_datasets()


if __name__ == '__main__':
    register_defaults()
