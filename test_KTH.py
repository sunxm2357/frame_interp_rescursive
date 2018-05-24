import numpy as np
import cv2
import math
import numpy
import torch
import torch.utils.serialization
import PIL
import PIL.Image
from dataloader.datasets import *
from dataloader.options import *
from run import *
import skimage.measure as measure
from skimage.measure import compare_ssim as ssim

torch.cuda.device(1)  # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


def metrics(seq_batch, opt, true_data, pred_data,  psnr_err, ssim_err, multichannel=True):
    pred_data = np.concatenate((seq_batch[:, :, :, :opt.K], pred_data, seq_batch[:, :, :, opt.K + opt.T:]),  axis=3)
    true_data = np.concatenate((seq_batch[:, :, :, :opt.K], true_data, seq_batch[:, :, :, opt.K + opt.T:]), axis=3)
    seq_len = opt.K + opt.T + opt.F

    cpsnr = np.zeros((seq_len,))
    cssim = np.zeros((seq_len,))

    for t in range(seq_len):
        pred = pred_data[:, :, :, t].astype("uint8")
        target = true_data[:, :, :, t].astype("uint8")
        if not multichannel:
            pred = np.squeeze(pred, axis=-1)
            target = np.squeeze(target, axis=-1)
        cpsnr[t] = measure.compare_psnr(pred, target)
        cssim[t] = ssim(target, pred, multichannel=multichannel)
    psnr_err = np.concatenate((psnr_err, cpsnr[None, opt.K:opt.K + opt.T]), axis=0)
    ssim_err = np.concatenate((ssim_err, cssim[None, opt.K:opt.K + opt.T]), axis=0)
    return psnr_err, ssim_err


def recursive_inpainting(first_img, last_img, T):
    [h, w, c] = first_img.shape
    pred = np.zeros((h, w, c, T + 2))
    pred[:, :, :, 0] = first_img
    pred[:, :, :, -1] = last_img
    start = 0
    end = T + 1
    p_start = start
    p_end = [end]
    while len(p_end) != 0:
        middle = (p_start + p_end[-1]) / 2
        if p_start != middle:
            print('producing the %s frame' % middle)
            pred[:, :, :, middle] = single_pred(pred[:, :, :, p_start], pred[:, :, :, p_end])
            p_end.append(middle)
        else:
            p_start = middle
            p_end.pop(-1)
            continue
    return pred[:, :, :, ::-1]


def main():
    save_freq = 50
    opt = TestOptions().parse()
    if opt.data == "KTH":
        lims_ssim = [1, opt.T, 0.6, 1]
        lims_psnr = [1, opt.T, 20, 34]
    elif opt.data in ['UCF', 'HMDB51']:
        lims_ssim = [1, opt.T, 0.3, 1]
        lims_psnr = [1, opt.T, 10, 35]
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.data)
    psnr_err = np.zeros((0, opt.T))
    ssim_err = np.zeros((0, opt.T))

    dataset = CreateDataset(opt)
    dataset_size = len(dataset)
    print('# testing videos = %d' % dataset_size)
    for i in range(dataset_size):
        print('dealing %d/%d' % (i, dataset_size))
        datas = dataset[i]
        if opt.pick_mode == 'First': datas = [datas]
        for data in datas:
            seq_batch = data['targets']
            last_preceding = seq_batch[:, :, :, opt.K-1]
            first_following = seq_batch[:, :, :, opt.K + opt.T]
            pred_data = recursive_inpainting(last_preceding, first_following, opt.T)
            true_data = seq_batch[:, :, :, opt.K: opt.K + opt.T]
            if opt.data == 'KTH':
                true_data = bgr2gray_np(true_data)
                pred_data = bgr2gray_np(pred_data)
            psnr_err, ssim_err = metrics(seq_batch, opt, true_data, pred_data, psnr_err, ssim_err, multichannel=(opt.data != 'KTH'))

        if i % (save_freq) == 0 or i == (dataset_size-1):
            print('psnr:', psnr_err.mean(axis=0))
            print('ssim:', ssim_err.mean(axis=0))
            save_path = os.path.join(opt.quant_dir, 'results_model.npz')
            np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
            psnr_plot = os.path.join(opt.quant_dir, 'psnr_final.png')
            draw_err_plot(psnr_err, 'Peak Signal to Noise Ratio', path=psnr_plot, lims=lims_psnr, type="Test")
            ssim_plot = os.path.join(opt.quant_dir, 'ssim_final.png')
            draw_err_plot(ssim_err, 'Structural Similarity', path=ssim_plot, lims=lims_ssim, type="Test")
    print("Done.")


if __name__ == "__main__":
    main()
