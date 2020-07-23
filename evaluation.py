import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from auxiliary.utils import AverageValueMeter, get_pred_from_cls_output, rotation_err, rotation_acc


def val(data_loader, model, bin_size, cls_data):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # load data and label
            im, shapes, label = data
            im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()

            # forward pass
            out = model(im, shapes, mean_class_data=cls_data)

            # transform the output into the label format
            preds = get_pred_from_cls_output([out[0], out[1], out[2]])
            for n in range(len(preds)):
                pred_delta = out[n + 3]
                delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
                preds[n] = (preds[n].float() + delta_value + 0.5) * bin_size
            pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1)

            predictions.append(pred)
            labels.append(label)

    predictions = torch.cat(predictions, 0)
    labels = torch.cat(labels, 0)

    return predictions, labels


def test_category(dataset_test, model, bin_size, cls, predictions_path, logname, cls_data):

    # initialize data loader and run validation
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    predictions, labels = val(test_loader, model, bin_size, cls_data)

    # calculate the rotation errors between prediction and ground truth
    test_errs = rotation_err(predictions, labels.float()).cpu().numpy()
    Acc = 100. * np.mean(test_errs <= 30)
    Med = np.median(test_errs)

    # save results
    outfile = os.path.join(predictions_path, 'results_{}.npz'.format(cls))
    np.savez(outfile, preds=predictions.cpu().numpy(), labels=labels.cpu().numpy(), errors=test_errs)

    with open(logname, 'a') as f:
        f.write('test accuracy for %d images of category %s \n' % (len(dataset_test), cls))
        f.write('Med_Err is %.2f, and Acc_pi/6 is %.2f \n \n' % (Med, Acc))

    return Acc, Med, test_errs
