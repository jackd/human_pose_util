import json
import numpy as np
from human_pose_util.register import get_dataset
from human_pose_util.dataset.normalize import normalize_dataset
from human_pose_util.dataset.normalize import filter_dataset
from human_pose_util.dataset.group import copy_group


class ReportManager(object):
    def __init__(self, key, summarize_fn, category_fn=None):
        if category_fn is None:
            def category_fn(x):
                return x
        self._category_fn = category_fn
        self._summarize_fn = summarize_fn
        self._key = key

    def category(self, sequence):
        return self._category_fn(sequence)

    def summarize(self, sequence, result):
        return self._summarize_fn(sequence, result)

    @property
    def key(self):
        return self._key

    def report(self, results, overwrite=False):
        with open(results.attrs['params_path'], 'r') as f:
            params = json.load(f)
        dataset = get_dataset(params['dataset']['type'])
        dataset = filter_dataset(
            dataset, modes=['eval'], **params['dataset']['filter_kwargs'])
        dataset = copy_group(dataset)
        dataset = normalize_dataset(
            dataset, **params['dataset']['normalize_kwargs'])
        summaries = {}
        for k, result in results.items():
            sequence = dataset[k]
            cat = self.category(sequence)
            if self.key in result and not overwrite:
                summary = result[self.key]
            else:
                print('Generating %s: %s' % (self.key, k))
                summary = self.summarize(sequence, result)
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


def proc_summary(sequence, result):
    from human_pose_util.evaluate import procrustes_error
    err = procrustes_error(sequence['p3w'], sequence['p3w'])
    return err * sequence.attrs['space_scale']


def sequence_proc_summary(sequence, result):
    from human_pose_util.evaluate import sequence_procrustes_error
    err = sequence_procrustes_error(
        np.array(sequence['p3w']), np.array(result['p3w']))
    return err * sequence.attrs['space_scale']
