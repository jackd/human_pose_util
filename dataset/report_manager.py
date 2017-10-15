import numpy as np
from human_pose_util.register import dataset_register


class ReportManager(object):
    def __init__(self, key, summarize_fn, category_fn=None):
        if category_fn is None:
            def category_fn(x):
                return x
        self._category_fn = category_fn
        self._summarize_fn = summarize_fn
        self._key = key

    def category(self, example):
        return self._category_fn(example)

    def summarize(self, example, result, dataset_attrs):
        return self._summarize_fn(example, result, dataset_attrs)

    @property
    def key(self):
        return self._key

    def report(self, results, overwrite=False):
        dataset = dataset_register[results.attrs['dataset']]['eval']
        dataset_attrs = dataset.attrs
        summaries = {}
        for k, example in dataset.items():
            result = results[k]
            cat = self.category(example)
            if self.key in result and not overwrite:
                summary = result[self.key]
            else:
                print('Generating %s: %s' % (self.key, k))
                summary = self.summarize(example, result, dataset_attrs)
                if hasattr(result, 'create_dataset'):
                    if self.key in result:
                        assert(overwrite)
                        # assert(hasattr(result, '__del__'))
                        del result[self.key]
                    result.create_dataset(self.key, data=summary)
            if cat not in summaries:
                summaries[cat] = []
            summaries[cat].append(summary)
        print('Results for %s' % self.key)
        summaries = {
            k: np.concatenate(v, axis=0) for k, v in summaries.items()}
        for k, summary in summaries.items():
            summary = np.mean(summary)
            print('%s: %.2f' % (k, summary))
        summary = np.concatenate([v for k, v in summaries.items()], axis=0)
        summary = np.mean(summary)
        print('Total: %.2f' % summary)


def proc_summary(example, result, dataset_attrs):
    from human_pose_util.evaluate import procrustes_error
    err = procrustes_error(example['p3w'], result['p3w'])
    return err * dataset_attrs['space_scale']


def sequence_proc_summary(example, result, dataset_attrs):
    from human_pose_util.evaluate import sequence_procrustes_error
    err = sequence_procrustes_error(
        np.array(example['p3w']), np.array(result['p3w']))
    return err * dataset_attrs['space_scale']
