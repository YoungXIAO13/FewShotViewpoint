import argparse
import os
from tqdm import tqdm
import numpy as np
import collections


parser = argparse.ArgumentParser()
parser.add_argument('result_head', type=str, default=None)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

test_mode = 'pred_novel'

acc = collections.defaultdict(list)
err = collections.defaultdict(list)

errors_all = collections.defaultdict(list)

times = np.arange(1, args.runs + 1).tolist()

mean_acc = []
mean_err = []
for i in tqdm(times):
    result_dir = os.path.join('{}_run{}'.format(args.result_head, i), test_mode)
    test_classes = sorted(
        [name.split('.')[0].replace('results_', '') for name in os.listdir(result_dir) if name.endswith('.npz')])

    for cls in test_classes:
        errors = np.load(os.path.join(result_dir, 'results_{}.npz'.format(cls)))['errors']
        acc[cls].append(np.mean(errors <= 30))
        err[cls].append(np.median(errors))

        errors_all[i] = errors_all[i] + errors.tolist()

    mean_acc.append(np.mean([acc[cls][i - 1] for cls in acc.keys()]))
    mean_err.append(np.mean([err[cls][i - 1] for cls in acc.keys()]))

# perf for each cls averaged over multiple runs
for cls in acc.keys():
    print('Class {}: Acc is {:.2f}; MedErr is {:.1f}'.format(cls, np.mean(acc[cls]), np.mean(err[cls])))

# perf across the whole dataset averaged over multiple runs
all_acc = [np.mean(np.array(errors_all[i]) <= 30) for i in errors_all.keys()]
all_err = [np.median(errors_all[i]) for i in errors_all.keys()]
print('Acc is {:.2f} +- {:.3f} || Err is {:.1f} +- {:.2f}'.format(
    np.mean(all_acc), np.std(all_acc), np.mean(all_err), np.std(all_err)))
