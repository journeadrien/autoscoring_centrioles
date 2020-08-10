# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:18:21 2020

@author: journe
"""
from __future__ import print_function


import numpy as np
import scipy.misc
import cv2
from glob import glob
import skimage.external.tifffile as tf
import xml.etree.ElementTree as ET
import re
import os.path as op

from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist
import errno
import os

class Image_Reader:
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.channels = ['DAPI','GFP','RFP','Cy5']
        self.mode = self.find_mode()
        self.list_file = self.find_file()
        self.z_stack, self.z_res, self.channel_order = self.read_metadata()
        self.pos = self.find_pos()
        self.meta_channel = self.get_marker()
        self.range_z = self.get_z_stack()
    
    def find_mode(self):
        lst_dir = list(os.listdir(self.folder_path))
        if any([x.endswith('.ome.tif') for x in lst_dir]):
            return 'ome'
        return 'folder'
    
    def find_file(self):
        if self.mode == 'ome':
            list_file = os.listdir(self.folder_path)
            return [x for x in list_file if x.endswith('.ome.tif')]
        if self.mode == 'folder':
            return next(os.walk(self.folder_path))[1]
    
    def read_metadata(self):
        if self.mode == 'ome':
            with tf.TiffFile(op.join(self.folder_path,self.list_file[0]),pages =0) as tif:
                z_stack = int(tif.micromanager_metadata['summary']['Slices'])
                z_res = round(float(tif.micromanager_metadata['summary']['z-step_um']),3)
                channel_order_str = tif.micromanager_metadata['summary']['ChNames']
                channel_order = {x.split(' - ')[1]:int(x.split(' - ')[0]) for x in channel_order_str}
        if self.mode == 'folder':
            attrib = self.parse_acquisition()
            z_res = round(float(attrib['acqZstep']),3)
            z_stack = int((float(attrib['acqZtop'])- float(attrib['acqZbottom'])) / z_res)+1
            channel_order_str = [value for key, value in attrib.items() if key.startswith('acqChannelName')]
            channel_order = {x.split(' - ')[1]:int(x.split(' - ')[0]) for x in channel_order_str}
        return z_stack, z_res, channel_order
    
    
    def parse_acquisition(self):
        tree =  ET.parse(glob(op.join(self.folder_path,'*_Acquisitio-Settings.txt'))[0])
        d ={}
        for p in tree.getiterator():
            if p.tag == 'entry':
                d[p.attrib['key']] = p.attrib['value']
        return d           
    
    def find_pos(self):
        pos ={}
        for file in self.list_file:
            pos_file = re.findall('\d{3}_\d{3}',file)[-1]
            x,y = pos_file.split('_')
            pos[pos_file] = {'x': int(x), 'y': int(y), 'file_path':op.join(self.folder_path,file)}
        return pos
    
    def get_z_stack(self):
        range_z = 66
        focus = int(self.z_stack/2)
        start = max(0,int(focus - range_z*self.z_res/2))
        stop = min(self.z_stack,int(focus+2 + range_z*self.z_res/2))
        return np.array(range(start,stop))
    
    def get_image(self,pos,channel):
        path_file = self.pos[pos]['file_path']
        if self.mode == 'ome':
            index_ch = self.meta_channel[channel]['index']
            pages = list((index_ch-1) * self.z_stack + self.range_z)
            with tf.TiffFile(op.join(self.folder_path,path_file),pages = pages) as tif:
                img = tif.asarray()
        if self.mode == 'folder':
            path_list = [self.get_path_image(path_file,channel,z) for z in self.range_z]
            img =[]
            for file in path_list:
                with tf.TiffFile(file) as tif:
                    img.append(tif.asarray())
            img = np.array(img)
        return img
            
              
    def get_path_image(self,path,channel,z):
        z_str = ('' if z>99 else '0' if z>9 else '00') + str(z)
        return glob(op.join(path,'*'+channel+'*'+str(z_str)+'*.tif'))[0]
    
    def get_marker(self):
        _,folder_name = op.split(self.folder_path)
        info_exp = folder_name.split('_')
        if len(info_exp) == 4:
            cell_line, treatment, markers, nb_exp = info_exp
        elif len(info_exp) == 3:
            cell_line, markers, nb_exp = info_exp
        else:
            raise 'Format Exp name not recognise'
        markers = markers.split('+')
        
        if 'DAPI' not in markers:
            markers.insert(0,'DAPI')
        meta_channel = {}
        for ch, mar in zip(self.channels[:len(markers)], markers):
            meta_channel[ch] = {'name':mar,'index':self.channel_order[ch]}
        return meta_channel
            

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))

def DAPI_image(images,how = 'maximun projection'):
    if how == 'maximum projection':
        return np.max(images, axis=0)
    elif how == 'focus':
        return images[int(images.shape[0]/2)]
    
def load_img(fname, color='RGB'):
    img = cv2.imread(fname,cv2.IMREAD_ANYDEPTH)
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img  

def get_image(path,channel,z_stack,color):
    z_stack = ('' if z_stack>99 else '0' if z_stack>9 else '00') + str(z_stack)
    return load_img(glob(path+'*'+channel+'*'+str(z_stack)+'*.tif')[0], color)

def get_images(path,channels,z_stacks,color):
    if isinstance(channels, str):
        return np.array([get_image(path,channels,z_stack,color) for z_stack in z_stacks])
    if isinstance(z_stacks, int):
        return {channel : get_image(path,channel,z_stacks) for channel in channels}
    return {channel : np.array([get_image(path,channel,z_stack,color) for z_stack in z_stacks]) for channel in channels}


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = cv2.resize(
            image, (round(w * scale),round(h * scale)),interpolation=cv2.INTER_CUBIC)
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

