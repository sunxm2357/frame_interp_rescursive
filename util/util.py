from __future__ import print_function
import torch
import io
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def print_current_errors(log_name, update, errors, t):
    message = 'update: %d, time: %.3f ' % (update, t)
    for k, v in errors.items():
        if k.startswith('Update'):
            message += '%s: %s ' % (k, str(v))
        else:
            message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, 'a') as log_file:
        log_file.write('%s \n' % message)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    """
    :param image_tensor: [batch_size, c, h, w]
    :param imtype: np.uint8
    :return: ndarray [batch_size, c, h, w]
    """
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[1] == 1:
        image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def tensorlist2imlist(tensors):
    ims = []
    for tensor in tensors:
        ims.append(tensor2im(tensor.data))
    return ims


def inverse_transform(images):
    return (images+1.)/2


def fore_transform(images):
    return images * 2 - 1


def bgr2gray(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.2989 * image[:, 2, :, :]
    gray = torch.unsqueeze(gray_, 1)
    return gray


def bgr2gray_np(image):
    # rgb -> grayscale 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_ = 0.1140 * image[:, :, 0, :] + 0.5870 * image[:, :, 1, :] + 0.2989 * image[:, :, 2, :]
    gray = np.expand_dims(gray_, axis=2)
    return gray


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def draw_frame(img, is_input):
    if img.shape[2] == 1:
        img = np.repeat(img, [3], axis=2)

    if is_input:
        img[:2,:,0]  = img[:2,:,2] = 0
        img[:,:2,0]  = img[:,:2,2] = 0
        img[-2:,:,0] = img[-2:,:,2] = 0
        img[:,-2:,0] = img[:,-2:,2] = 0
        img[:2,:,1]  = 255
        img[:,:2,1]  = 255
        img[-2:,:,1] = 255
        img[:,-2:,1] = 255
    else:
        img[:2,:,0]  = img[:2,:,1] = 0
        img[:,:2,0]  = img[:,:2,1] = 0
        img[-2:,:,0] = img[-2:,:,1] = 0
        img[:,-2:,0] = img[:,-2:,1] = 0
        img[:2,:,2]  = 255
        img[:,:2,2]  = 255
        img[-2:,:,2] = 255
        img[:,-2:,2] = 255
    return img


def draw_frame_tensor(img, K, T):
    img[:, 0, :2, :] = img[:, 2, :2, :] = 0
    img[:, 0, :, :2] = img[:, 2, :, :2] = 0
    img[:, 0, -2:, :] = img[:, 2, -2:, :] = 0
    img[:, 0, :, -2:] = img[:, 2, :, -2:] = 0
    img[:, 1, :2, :] = 1
    img[:, 1, :, :2] = 1
    img[:, 1, -2:, :] = 1
    img[:, 1, :, -2:] = 1
    img[K:K+T, 0, :2, :] = img[K:K+T, 1, :2, :] = 0
    img[K:K+T, 0, :, :2] = img[K:K+T, 1, :, :2] = 0
    img[K:K+T, 0, -2:, :] = img[K:K+T, 1, -2:, :] = 0
    img[K:K+T, 0, :, -2:] = img[K:K+T, 1, :, -2:] = 0
    img[K:K+T, 2, :2, :] = 1
    img[K:K+T, 2, :, :2] = 1
    img[K:K+T, 2, -2:, :] = 1
    img[K:K+T, 2, :, -2:] = 1
    return img


def draw_err_plot(err,  err_name, lims, path=None, type="Val"):
    avg_err = np.mean(err, axis=0)
    T = err.shape[1]
    fig = plt.figure()
    # plt.clf()
    ax = fig.add_subplot(111)
    x = np.arange(1, T+1)
    ax.plot(x, avg_err, marker="d")
    ax.set_xlabel('time steps')
    ax.set_ylabel(err_name)
    ax.grid()
    ax.set_xticks(x)
    ax.axis(lims)
    if type == 'Val':
        plot_buf = gen_plot(fig)
        im = np.array(Image.open(plot_buf), dtype=np.uint8)
        plt.close(fig)
        return im
    elif type == 'Test':
        plt.savefig(path)
    else:
        raise ValueError('error plot type [%s] is not defined' % type)


def plot_to_image(x, y, lims):
    '''
    Plot y vs. x and return the graph as a NumPy array
    :param x: X values
    :param y: Y values
    :param lims: [x_start, x_end, y_start, y_end]
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.axis(lims)
    plot_buf = gen_plot(fig)
    im = np.array(Image.open(plot_buf), dtype=np.uint8)
    im = np.expand_dims(im, axis=0)
    plt.close(fig)
    return im


def gen_plot(fig):
    """
    Create a pyplot plot and save to buffer.
    https://stackoverflow.com/a/38676842
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visual_grid(seq_batch, pred, K, T, infill=False, pred_forward=None, pred_backward=None):
    pred_data = torch.stack(pred, dim=-1)
    vis_f = False
    vis_b = False
    if not pred_forward is None:
        pred_forward_data = torch.stack(pred_forward, dim=-1)
        vis_f = True
    if not pred_backward is None:
        pred_backward_data = torch.stack(pred_backward, dim=-1)
        vis_b = True
    true_data = seq_batch[:, :, :, :, K:K + T].clone()
    if not infill:
        pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data], dim=-1)
        true_data = torch.cat([seq_batch[:, :, :, :, :K], true_data], dim=-1)
    else:
        if vis_f:
            pred_forward_data = torch.cat([seq_batch[:, :, :, :, :K], pred_forward_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
        if vis_b:
            pred_backward_data = torch.cat([seq_batch[:, :, :, :, :K], pred_backward_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
        pred_data = torch.cat([seq_batch[:, :, :, :, :K], pred_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
        true_data = torch.cat([seq_batch[:, :, :, :, :K], true_data, seq_batch[:, :, :, :, K+T:]], dim=-1)
    batch_size = int(pred_data.size()[0])
    c_dim = int(pred_data.size()[1])
    seq_len = int(pred_data.size()[-1])
    vis = []
    for i in range(batch_size):
        if vis_f:
            pred_forward_data_sample = inverse_transform(pred_forward_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        if vis_b:
            pred_backward_data_sample = inverse_transform(pred_backward_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        pred_data_sample = inverse_transform(pred_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        target_sample = inverse_transform(true_data[i, :, :, :, :]).permute(3, 0, 1, 2)
        if c_dim == 1:
            if vis_f:
                pred_forward_data_sample = torch.cat([pred_forward_data_sample] * 3, dim=1)
            if vis_f:
                pred_backward_data_sample = torch.cat([pred_backward_data_sample] * 3, dim=1)
            pred_data_sample = torch.cat([pred_data_sample] * 3, dim=1)
            target_sample = torch.cat([target_sample] * 3, dim=1)
        if vis_f:
            pred_forward_data_sample = draw_frame_tensor(pred_forward_data_sample, K, T)
        if vis_b:
            pred_backward_data_sample = draw_frame_tensor(pred_backward_data_sample, K, T)
        pred_data_sample = draw_frame_tensor(pred_data_sample, K, T)
        target_sample = draw_frame_tensor(target_sample, K, T)
        draw_sample = []
        if vis_f:
            draw_sample.append(pred_forward_data_sample)
        if vis_b:
            draw_sample.append(pred_backward_data_sample)
        draw_sample += [pred_data_sample, target_sample]
        output = torch.cat(draw_sample, dim=0)
        vis.append(vutils.make_grid(output, nrow=seq_len))
    grid = torch.cat(vis, dim=1)
    grid = torch.from_numpy(np.clip(np.flip(grid.numpy(), 0).copy(), 0, 1))
    return grid


def opt2model(opt):
    model = None
    if opt.model == 'kernelcomb' and opt.rc_loc != -1:
        model = 'TAI'

    if opt.model == 'kernelcomb' and opt.rc_loc == -1 and opt.comb_type == 'w_avg':
        model = 'TWI'

    if opt.model == 'simplecomb' and opt.comb_type == 'w_avg':
        model = 'bi-TW'

    if opt.model == 'simplecomb' and opt.comb_type == 'avg':
        model = 'bi-SA'

    if opt.model == 'mcnet':
        model = 'MC-Net'

    if model is None:
        print('{model, rc_loc, comb_type} = {%s, %d, %s} does not have a name'%(opt.model, opt.rc_loc, opt.comb_type))
        model = 'Unknown'

    return model


def opt2model_trivial(opt):
    model = None
    if opt.comb_type == 'repeat_P' or opt.comb_type == 'repeat_F':
        model = opt.comb_type

    if opt.comb_type == 'w_avg':
        model = 'TW_P_F'

    if opt.comb_type == 'avg':
        model = 'SA_P_F'

    if model is None:
        print('comb_type = {%s} does not have a name for trivial baseline'%(opt.comb_type))
        model = 'Unknown'

    return model


def refresh_donelist(opt, is_trivial=False):
    makedir('records/')
    exps_path = 'records/finished_exp.npy'
    data_name = {'KTH': 'KTH Actions', 'UCF': 'UCF-101', 'HMDB51': 'HMDB-51'}
    data = data_name[opt.data]
    if is_trivial:
        model = opt2model_trivial(opt)
    else:
        model = opt2model(opt)
    i_o_num = '%d_%d'%(opt.K, opt.T)
    if os.path.exists(exps_path):
        exps_dict = np.load(exps_path).item()
    else:
        exps_dict={}

    if not data in exps_dict.keys():
        exps_dict[data] = {}

    if not model in exps_dict[data].keys():
        exps_dict[data][model] = {}

    if not i_o_num in exps_dict[data][model].keys():
        exps_dict[data][model][i_o_num] = []

    exps_dict[data][model][i_o_num].append(opt.test_name)

    np.save(exps_path, exps_dict)


def listopt(opt, f=None):
    args = vars(opt)

    if f is not None:
        f.write('------------ Options -------------\n')
    else:
        print('------------ Options -------------')

    for k, v in sorted(args.items()):
        if f is not None:
            f.write('%s: %s\n' % (str(k), str(v)))
        else:
            print('%s: %s' % (str(k), str(v)))

    if f is not None:
        f.write('-------------- End ----------------\n')
    else:
        print('-------------- End ----------------')


def to_numpy(tensor, gpu_ids, transpose=None):
    # turn tensor into numpy file
    if len(gpu_ids) > 0:
        arr = tensor.cpu().numpy()
    else:
        arr = tensor.numpy()
    if transpose is not None:
        arr = np.transpose(arr, transpose)

    return arr
