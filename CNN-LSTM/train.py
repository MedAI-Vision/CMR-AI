import os
import shutil
import mmcv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import SimpleITK as sitk
from frame import densenet121, densenet_40_12_bc
from sequence import RNN, MetaRNN, SeqSumPoolingEncoder

import time
import warnings
from torchvision import transforms
from PIL import Image
import random
import torchvision.transforms.functional as tf

from sklearn.model_selection import train_test_split
import argparse
import torch.distributed as dist
import pickle as pkl

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

rank = torch.distributed.get_rank()
setup_seed(1 + rank)

def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class mydataset(torch.utils.data.Dataset):
    def __init__(self, datapath, disease):

        with open(r'/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl', 'rb') as f:
            data_mask_map = pkl.load(f)
        f.close()

        with open(datapath, 'r') as f:
            self.sequence = f.readlines()
            random.shuffle(self.sequence)

        self.data = []
        self.labels = []

        self.num = 13

        # for i in range(len(self.sequence)):
        for i, element in enumerate(tqdm.tqdm(self.sequence)):
            path_contents = [c for c in self.sequence[i].replace('\n', '').split(' ')]
            class_id = int(path_contents[-1])
            image_dir = path_contents[0]
            self.labels.append(class_id)

            if not os.path.exists(image_dir):
                continue

            mask_path = data_mask_map[image_dir]
            data_ori = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
            data = np.zeros((self.num, data_ori.shape[1], data_ori.shape[2]))
            for i in range(self.num):
                data[i, :, :] = data_ori[i * 2, :, :]

            try:
                data = data[:, mask_path[2]:mask_path[3], mask_path[0]:mask_path[1]]
                data = self.pad(data, (210, 210))
                data = self.normalize(data, [154.5, 154.5, 154.5], [66.62, 66.62, 66.62])
                data = self.resize(data, (64, 64))

                data_up_ori = sitk.GetArrayFromImage(sitk.ReadImage(image_dir.replace('mid', 'up')))
                data_up = np.zeros((self.num, data_up_ori.shape[1], data_up_ori.shape[2]))
                for i in range(self.num):
                    data_up[i, :, :] = data_up_ori[i * 2, :, :]
                data_up = data_up[:, mask_path[2]:mask_path[3], mask_path[0]:mask_path[1]]
                data_up = self.pad(data_up, (210, 210))
                data_up = self.normalize(data_up, [154.5, 154.5, 154.5], [66.62, 66.62, 66.62])
                data_up = self.resize(data_up, (64, 64))

                data_down_ori = sitk.GetArrayFromImage(sitk.ReadImage(image_dir.replace('mid', 'down')))
                data_down = np.zeros((self.num, data_down_ori.shape[1], data_down_ori.shape[2]))
                for i in range(self.num):
                    data_down[i, :, :] = data_down_ori[i * 2, :, :]
                data_down = data_down[:, mask_path[2]:mask_path[3], mask_path[0]:mask_path[1]]
                data_down = self.pad(data_down, (210, 210))
                data_down = self.normalize(data_down, [154.5, 154.5, 154.5], [66.62, 66.62, 66.62])
                data_down = self.resize(data_down, (64, 64))

                np_array = [data_up, data, data_down]
                volume = np.moveaxis(np_array, 0, 1)
                # batch_size, num_frames, num_channels, width, height
                self.data.append((volume, class_id))

            except:
                pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('mydata: ', self.data[index][1])
        return self.data[index]

    def pad(self, img, size=(172, 172)):
        img = img.transpose(1, 2, 0)
        x = img.shape[0]
        y = img.shape[1]
        pad_x1 = (size[0] - x) // 2
        pad_x2 = size[0] - x - pad_x1
        pad_y1 = (size[1] - y) // 2
        pad_y2 = size[1] - y - pad_y1
        new_img = np.zeros((size[0], size[1], img.shape[-1]))
        if pad_x1 < 0 or pad_x2 < 0:
            img = img[-pad_x1:x + pad_x2, :, :]
            pad_x1 = 0
            pad_x2 = 0
        if pad_y1 < 0 or pad_y2 < 0:
            img = img[:, -pad_y1:y + pad_y2, :]
            pad_y1 = 0
            pad_y2 = 0
        for i in range(img.shape[-1]):
            new_img[:, :, i] = np.pad(img[:, :, i], ((pad_x1, pad_x2), (pad_y1, pad_y2)), 'constant',
                                      constant_values=(0, 0))
        new_img = new_img.transpose(2, 0, 1)

        return new_img

    def resize(self, data, size):
        ch = data.shape[0]
        new_img = np.zeros((ch, size[0], size[1]))
        for i in range(ch):
            img = data[i, :, :]
            new_img[i, :, :] = cv2.resize(img, (size[0], size[1]))
        return new_img

    def normalize(self, data, mean, std):
        ch = data.shape[0]
        
        # single norm
        mean = np.mean(data)
        std = np.std(data)

        new_img = np.zeros((ch, data.shape[1], data.shape[2]))
        for i in range(ch):
            img = data[i, :, :]
            img = np.expand_dims(img, 2).repeat(3, axis=2)

            mmcv.imnormalize(img, np.array(mean), np.array(std))
            new_img[i, :, :] = img[:, :, 0]

        return new_img

    def hist_normalization(self, img, a=0, b=255):
        # get max and min
        c = np.min(img)
        d = np.max(img)
        if c == d:
            return img

        # normalization
        img = (b - a) / (d - c) * (img - c) + a
        img[img < a] = a
        img[img > b] = b
        img = img.astype(np.uint8)
        return img

    def hist_equalization(self, img_array):
        histogram_array = np.bincount(img_array.astype(int).flatten(), minlength=256)
        # normalize
        histogram_array = histogram_array / np.sum(histogram_array)
        # cumulative histogram
        chistogram_array = np.cumsum(histogram_array)
        """
        STEP 2: Pixel mapping lookup table
        """
        transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
        """
        STEP 3: Transformation
        """
        # flatten image array into 1D list
        img_list = list(img_array.flatten())
        # transform pixel values to equalize
        eq_img_list = [transform_map[p] for p in img_list]
        # reshape and write back into img_array
        eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

        return eq_img_array


class MRISequenceNet(nn.Module):
    """
    Simple container network for MRI sequence classification. This module consists of:

        1) A frame encoder, e.g., a ConvNet/CNN
        2) A sequence encoder for merging frame representations, e.g., an RNN

    """

    def __init__(self, frame_encoder, seq_encoder, use_cuda=False):
        super(MRISequenceNet, self).__init__()
        self.fenc = frame_encoder
        self.senc = seq_encoder
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        return self.senc.init_hidden(batch_size)

    def embedding(self, x, hidden):
        """Get learned representation of MRI sequence"""
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        x = self.senc.embedding(x, hidden)
        if self.use_cuda:
            return x.cpu()
        else:
            return x

    def forward(self, x, hidden=None):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        # collapse all frames into new batch = batch_size * num_frames
        batch_size, num_frames, num_channels, width, height = x.size()
        # print('input: ', x.size()) [4, 13, 3, 64, 64]
        x = x.view(-1, num_channels, width, height)
        # encode frames
        # print('input_view: ',x.size())   [52, 3, 64, 64]
        x = self.fenc(x)
        # print('fenc output: ',x.size())    [52, 132, 1, 1]
        x = x.view(batch_size, num_frames, -1)
        # print('fenc output view: ',x.size()) [2, 13, 132]
        # encode sequence
        x = self.senc(x, hidden)
        # print('senc output: ',x.size()) [2, 2]
        return x

    def predict_proba(self, data_loader, binary=True, pos_label=1):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = Variable(x) if not self.use_cuda else Variable(x).cuda()
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x.size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)
        return y_pred[:, pos_label] if binary else y_pred

    def predict(self, data_loader, binary=True, pos_label=1, threshold=0.5, return_proba=False, topSelection=None):
        """
        If binary classification, use threshold on positive class
        If multinomial, just select the max probability as the predicted class
        :param data_loader:
        :param binary:
        :param pos_label:
        :param threshold:
        :return:
        """
        proba = self.predict_proba(data_loader, binary, pos_label)
        if topSelection is not None and topSelection < proba.shape[0]:
            threshold = proba[np.argsort(proba)[-topSelection - 1]]
        if binary:
            pred = np.array([1 if p > threshold else 0 for p in proba])
        else:
            pred = np.argmax(proba, 1)

        if return_proba:
            return (proba, pred)
        else:
            return pred


class Dense4012FrameRNN(MRISequenceNet):
    def __init__(self, n_classes, use_cuda, **kwargs):
        super(Dense4012FrameRNN, self).__init__(frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "Dense4012FrameRNN"
        input_shape = kwargs.get("input_shape", (3, 64, 64))

        seq_output_size = kwargs.get("seq_output_size", 128)
        seq_dropout = kwargs.get("seq_dropout", 0.1)
        seq_attention = kwargs.get("seq_attention", True)
        seq_bidirectional = kwargs.get("seq_bidirectional", True)
        seq_max_seq_len = kwargs.get("seq_max_seq_len", 13)
        seq_rnn_type = kwargs.get("rnn_type", "LSTM")
        pretrained = kwargs.get("pretrained", True)
        requires_grad = kwargs.get("requires_grad", True)

        self.fenc, _ = densenet_40_12_bc(pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)
        # print(input_shape)
        # print(frm_output_size)

        self.senc = RNN(n_classes=n_classes, input_size=frm_output_size, hidden_size=seq_output_size,
                        dropout=seq_dropout, max_seq_len=seq_max_seq_len, attention=seq_attention,
                        rnn_type=seq_rnn_type, bidirectional=seq_bidirectional, use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0, 1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size = self.fenc.forward(x).view(-1).size()[0]
        return frm_output_size


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:
            # print(len(self.dataset), index)
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]  # label
        # print('map: ', y)
        return x, y

    def __len__(self):
        return len(self.dataset)


def transform_train(image):
    angle = transforms.RandomRotation.get_params([-90, 90])
    flip = False
    cont = False
    negative = False
    if random.random() > 0.5:
        flip = True
    if random.random() > 0.5:
        cont = True
    if random.random() > 0.5:
        negative = True
    delta = torch.rand(1)
    if negative:
        delta = -delta
    delta = delta / 10
    for i in range(image.shape[0]):
        img = Image.fromarray(np.uint8(np.transpose(image[i, :, :, :], (1, 2, 0))))
        img = tf.rotate(img, angle)
        if cont:
            img = tf.adjust_contrast(img, 2)
        # if flip:
        #     img = tf.hflip(img)
        img = tf.to_tensor(img)
        # img = tf.normalize(img, [0.13078083, 0.13036211, 0.13005716], [0.23886861, 0.23799002, 0.23759232])
        image[i, :, :, :] = img + delta
    return image


def transform_test(image):
    for i in range(image.shape[0]):
        img = Image.fromarray(np.uint8(np.transpose(image[i, :, :, :], (1, 2, 0))))
        img = tf.to_tensor(img)
        # img = tf.normalize(img, [0.13268289, 0.13221595, 0.13187733], [0.24014601, 0.23915084, 0.23863715])
        image[i, :, :, :] = img
    return image


def run(disease, train_sax, test_sax):
    # define the datasets
    train_dataset = mydataset(train_sax, disease)
    test_dataset = mydataset(test_sax, disease)
    train_dataset = MapDataset(train_dataset, transform_train)
    test_dataset = MapDataset(test_dataset, transform_test)

    # define the samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=1, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=1, sampler=test_sampler)

    device = torch.device("cuda", args.local_rank)

    # choose Dense4012FrameRNN as our model (CNN-LSTM)
    model = Dense4012FrameRNN(2, True)

    # optimizer: SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # train with Multi-GPU
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

    epoch_resume = 0
    out_csv_name = 'log_cnn_lstm.csv'
    output_path = './output/cnn_lstm/'

    try:
        # trying to load checkpoint
        checkpoint = torch.load(output_path + 'pretrained_densenet_4012BC.pth', map_location=device)
        model.load_state_dict(checkpoint['net'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_resume = checkpoint["epoch"] + 1
        if args.local_rank == 0:
            print("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        if args.local_rank == 0:
            print("Starting to run from scratch\n")

    update_freq = 1
    use_cuda = True
    num_frames = None
    n_epochs = 200

    # define list to record loss and acc
    train_acc = 0.0
    test_acc = 0.0
    best_acc = 0.0
    if (not os.path.exists(output_path + out_csv_name)) & epoch_resume == 0:
        df = pd.DataFrame(columns=["epoch", "train loss", 'test loss', "train acc", "test acc"])
        df.to_csv(output_path + out_csv_name, index=False)
    start_time = time.time()

    # train loop
    for epoch in range(epoch_resume, n_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        train_loss, correct, total = 0.0, 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.type(torch.FloatTensor)
            if num_frames is not None:
                y = np.repeat(y, num_frames)
            if isinstance(x, list):
                x = [Variable(x_) if not use_cuda else Variable(x_).cuda(args.local_rank, non_blocking=True) for x_ in
                     x]
                h0 = model.module.init_hidden(x[0].size(0))
            else:
                x = Variable(x) if not use_cuda else Variable(x).cuda(args.local_rank, non_blocking=True)
                h0 = model.module.init_hidden(x.size(0))
            y = Variable(y) if not use_cuda else Variable(y).cuda(args.local_rank, non_blocking=True)
            outputs = model(x, h0)
            loss = criterion(outputs, y)
            optimizer.zero_grad()  # reset gradient
            loss.backward()
            optimizer.step()  # update parameters

            train_loss += loss.item()
            total += y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted.eq(y.data)).item()

        t = torch.tensor([correct, total, train_loss], dtype=torch.float64, device=device)
        dist.barrier()  # synchronizes all processes
        # Reduces the tensor data across all machines in such a way that all get the final result.
        dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM, )
        correct = t[0].item()
        total = t[1].item()
        train_loss = t[2].item()

        # progress update
        if (epoch + 1) % update_freq == 0 and args.local_rank == 0:
            elapsed = time.time() - start_time
            msg = 'Epoch: {:>8} | UT: {:>10} | ETA: {:>10} | Train Loss: {:0.3f} | Train Acc: {:>5}% ({:d}/{:d})'
            avg_loss = train_loss / (i + 1)
            acc = 100.0 * correct / total
            print(msg.format("[{}/{}]".format(epoch + 1, n_epochs), format_time(elapsed),
                             format_time(elapsed * (n_epochs - epoch_resume - epoch)),
                             avg_loss, "{:2.1f}".format(acc), int(correct), int(total)))
            train_loss = avg_loss
            train_acc = acc
            start_time = time.time()
        test_correct = 0
        test_total = 0
        test_loss = 0

        # evaluation
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x, y = batch
                x = x.type(torch.FloatTensor)
                if num_frames is not None:
                    y = np.repeat(y, num_frames)
                if isinstance(x, list):
                    x = [Variable(x_) if not use_cuda else Variable(x_).cuda(args.local_rank, non_blocking=True) for x_
                         in x]
                    h0 = model.module.init_hidden(x[0].size(0))
                else:
                    x = Variable(x) if not use_cuda else Variable(x).cuda(args.local_rank, non_blocking=True)
                    h0 = model.module.init_hidden(x.size(0))
                y = Variable(y) if not use_cuda else Variable(y).cuda(args.local_rank, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(x, h0)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_correct += predicted.eq(y.data).cpu().sum()
                test_total += y.size(0)
            t = torch.tensor([test_correct, test_total, test_loss], dtype=torch.float64, device=device)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(t,
                            op=torch.distributed.ReduceOp.SUM, )
            # Reduces the tensor data across all machines in such a way that all get the final result.
            test_correct = int(t[0].item())
            test_total = int(t[1].item())
            test_loss = t[2].item()
            if args.local_rank == 0:
                acc = test_correct / test_total
                avg_loss = test_loss / (i + 1)
                print(f'test accuracy: {acc} ({test_correct} / {test_total}) test loss: {avg_loss}')
                test_acc = acc
                test_loss = avg_loss
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.module.state_dict(),
                               output_path + 'best_cnnlstm_attn_sax_224.pth')
        if args.local_rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            torch.save(checkpoint, output_path + 'last_cnnlstm_attn_sax_224.pth')
        # scheduler.step()

        # save the log_per_epoch into log.csv
        df = pd.read_csv(output_path + out_csv_name)
        try:
            df.loc[epoch + 1] = [epoch, train_loss, test_loss, 0.01 * train_acc, test_acc]
            df.to_csv(output_path + out_csv_name, index=False)
        except:
            pass

train_data = './workdir/Screen_ann/sax_cine/0.994_fold_23.txt'
test_data = './workdir/Screen_ann/sax_cine/0.994_fold_1.txt'
run(['nor', 'abnor'], train_data, test_data)