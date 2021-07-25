import scipy.stats
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
#from ImageQualityDataset import ImageQualityDataset
# LQ add it
#from ImageQualityDatasetLQFull import ImageQualityDatasetLQFull
from ImageQualityDatasetLQSixSeperate import ImageQualityDatasetLQSixSeperate
from MeonLQ import Meon
from Gdn import Gdn
from PLCCLoss import PLCCLoss
from VideoTransforms import DenseSpatialCrop_collate
import re
from torch._six import string_classes, int_classes, FileNotFoundError
import collections
from torch.utils.data.dataloader import default_collate
import math
from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.output_channel = config.output_channel     # num of dist types
        self.use_cuda = torch.cuda.is_available() and config.use_cuda
        # to_tensor = transforms.ToTensor()

        self.enable_dist_test = True

        self.train_transform = transforms.Compose([transforms.CenterCrop(size=235), transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5), transforms.ToTensor()])
        #self.train_transform = transforms.Affine([transforms.Affine(??), transforms.ToTensor()])
        self.test_transform = lambda stride: transforms.Compose([DenseSpatialCrop_collate(output_size=235, stride=stride)])
        self.val_transform = self.test_transform

        self.train_batch_size = config.batch_size
        #self.train_data = ImageQualityDatasetLQ(csv_file=config.train_csv, root_dir=config.trainset, enable_dist=self.enable_dist_test, transform=self.train_transform)
        # LQ add it
        #self.train_data = ImageQualityDatasetLQFull(csv_file_dist=config.train_csv, csv_file_mos=config.train_csv_DT, root_dir_dist=config.trainsetDT, root_dir_mos=config.trainset, enable_dist=self.enable_dist_test, transform=self.test_transform, train=config.train)
        self.train_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=config.train_csv, csv_file_mos=config.train_csv_DT, root_dir_dist=config.trainsetDT, root_dir_mos=config.trainset, enable_dist=self.enable_dist_test, transform=self.test_transform, train=config.train)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,    #LQ chaange it from true to false
                                       pin_memory=True,
                                       num_workers=1)
        self.train_data_size = len(self.train_loader.dataset)
        self.num_steps_per_epoch = len(self.train_loader)
        self.resume = config.resume
        self.train = config.train


        if self.train:
            self.writer = SummaryWriter(log_dir=config.board)

        # val set configuration

        self.val_config = {
            "name": "test_PCMOS100",
            "num_workers": 1,
            "input_csv": "/home/zduanmu/QiLiu/MEONLQ/testset/mos/PCMeon_mos.txt",
            "root_dir": "/home/zduanmu/QiLiu/MEONLQ/testset/mos/",
            "use_cuda": torch.cuda.is_available() and config.use_cuda,
            "transform": self.val_transform,
            "save_path": "/home/zduanmu/QiLiu/MEONLQ/PC_test_results/PC_MOS100",
            "val_batch_size": 30
        }

        # initialize the model
        self.model = Meon(output_channel=self.output_channel)
        self.model_name = type(self.model).__name__
        print(self.model)

        # loss function
        self.crit_dist_type = nn.CrossEntropyLoss(size_average=True)
        self.crit_quality_pred = PLCCLoss() ###Loss is PLCC
       # self.crit_quality_pred = nn.L1Loss() ###Loss is L1
        #self.crit_quality_pred = nn.MSELoss()  ###Loss is L2
        #self.crit_quality_pred = nn.SmoothL1Loss()  ###Loss is L1/L2
        self.loss_dt = None
        self.loss_qp = None
        self.loss = None
        self.lamb = 0.025   # loss = entropy_loss + lambda * plcc_loss L1(0.025, 0.1 , 1, 10)
        #self.lamb = -2  # loss = entropy_loss + lambda * plcc_loss

        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
       # self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print("[*] GPU #", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()
            self.crit_dist_type = self.crit_dist_type.cuda()
            self.crit_quality_pred = self.crit_quality_pred.cuda()

        # some states
        # self.start_epoch = 0
        self.train_loss = []
        self.train_dt_loss = []
        self.train_qp_loss = []
        self.val_srcc = []
        self.val_plcc = []
        self.val_acc = []
        self.val_results = {}
        self.ckpt_path = config.ckpt_path
        self.ckpt = config.ckpt
        self.use_cuda = config.use_cuda
        self.max_epochs = config.max_epochs
        self.every_eval = config.every_eval
        self.epochs_per_save = config.epochs_per_save
        self.lr_scheduler_name = config.lr_scheduler

        self.pretrainDT = config.pretrainDT
        if self.pretrainDT:
            self.DT_ckpt_path = config.DT_ckpt_path
            self.DT_ckpt = config.DT_ckpt


        # load the model when resume or test
        if self.resume or not self.train:
            if self.ckpt:
                ckpt = os.path.join(self.ckpt_path, self.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)
            self.last_epoch = self.current_epoch
            last_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['initial_lr'] = last_lr
        # training
        else:
            self.last_epoch = -1

        # load the DT model when training the pretrainDT model
        if self.pretrainDT and self.train:
            if self.DT_ckpt:
                DT_ckpt = os.path.join(self.DT_ckpt_path, self.DT_ckpt)
            else:
                DT_ckpt = self._get_latest_checkpoint(path=config.DT_ckpt_path)
            self._load_DT_checkpoint(ckpt=DT_ckpt)

        # lr scheduler
        if self.lr_scheduler_name == "StepLR":
            self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                                 last_epoch=self.last_epoch,
                                                 step_size=config.decay_interval,
                                                 gamma=config.decay_ratio)
        elif self.lr_scheduler_name == "CosineAnnealingLR":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                            T_max=self.max_epochs * self.num_steps_per_epoch,
                                                            last_epoch=self.last_epoch)
        else:
            raise Exception("Wrong lr_scheduler_name")

    def patch_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        numpy_type_map = {
            'float64': torch.DoubleTensor,
            'float32': torch.FloatTensor,
            'float16': torch.HalfTensor,
            'int64': torch.LongTensor,
            'int32': torch.IntTensor,
            'int16': torch.ShortTensor,
            'int8': torch.CharTensor,
            'uint8': torch.ByteTensor,
        }
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
            tmp = torch.stack(batch, 0, out=out)
            return tmp
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))
                 # LQ: change it for dimension error
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
  #              return torch.cat([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], collections.Mapping):
            # return {key: dim1_collate([d[key] for d in batch]) for key in batch[0]}
            collated = {}
            collated["image"] = self.patch_collate([d["image"] for d in batch])
            collated["score"] = default_collate([d["score"] for d in batch])
            if self.enable_dist_test:
                collated["disttype"] = default_collate([d["disttype"] for d in batch])
            collated["image_name"] = default_collate([d["image_name"] for d in batch])
            collated["patch_num"] = default_collate([d["patch_num"] for d in batch])
            return collated

        raise TypeError((error_msg.format(type(batch[0]))))

    def fit(self):
        for epoch in range(self.last_epoch + 1, self.max_epochs):
            self._train_single_epoch(epoch)

    def _complete_config(self, config, num_workers=1, save_path="/home/zduanmu/Documents/WZL/MEON/tmp_results", transform=None):
        if 'input_csv' not in config:
            raise ValueError('Database file not specified!')

        if 'root_dir' not in config:
            raise ValueError('Database path not specified!')

        if 'name' not in config:
            config['name'], _ = os.path.splitext(os.path.basename(config['input_csv']))

        if 'num_workers' not in config:
            config['num_workers'] = num_workers

        if 'use_cuda' not in config:
            config['use_cuda'] = self.use_cuda

        """
        if 'transform' not in config:
            if transform is None:
                transform = self.val_transform
                # transform = transforms.Compose([transforms.RandomCrop(size=235), transforms.ToTensor()])
            config['transform'] = transform
        """

        if 'save_path' not in config:
            config['save_path'] = save_path

        if 'val_batch_size' not in config:
            config['val_batch_size'] = self.train_batch_size

        if 'test_batch_size' not in config:
            config['test_batch_size'] = self.train_batch_size

        return config

    """
    def _evaluateImage_single(self, eval_config):
        if eval_config is None:
            return None, None
        test_info = pd.read_csv(eval_config["input_csv"], header=None)
        image_names = []
        p = []
        q = []
        dq = []
        # p_hat = []
        q_hat = []
        dq_hat = []
        for idx, test_instance in test_info.iterrows():
            im_name = test_instance.iloc[0]
            image_dir = os.path.join(eval_config["root_dir"], im_name)
            score = test_instance.iloc[1]
            # disttype = test_instance.iloc[2]
            image_names.append(im_name)
            # p.append(disttype)
            q.append(score)

            if eval_config["use_cuda"]:
                self.model = self.model.cuda()

            if os.path.exists(image_dir):
                image_dir = [image_dir]  # image_dir must be a list
                pred_quality = predict_quality_image_NoDT(image_dir, self.model, eval_config)
                # p_hat.append(pred_disttype)
                q_hat.append(pred_quality)
                print('%s\t%.3f\t%.3f' % (test_instance.iloc[0], score, pred_quality))

                if not score in [100, 100.0]:
                    dq.append(score)
                    dq_hat.append(pred_quality)

        # acc = sum(np.equal(np.array(p_hat), np.array(p))) / len(p)
        srcc = scipy.stats.mstats.spearmanr(x=dq, y=dq_hat)[0]
        plcc = scipy.stats.mstats.pearsonr(x=dq, y=dq_hat)[0]

        print("SRCC: ", srcc, "PLCC: ", plcc)
        if eval_config['save_path'] is not None:
            save_path = os.path.join(eval_config['save_path'], self.model_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, self.model_name + '_' + str(self.start_epoch) + '_' + eval_config['name'] + '.pt')
            result = {'db_name': eval_config['name'],
                      'model_name': self.model_name,
                      'image_names': image_names,
                      # 'gt_disttype': p,
                      'gt_quality': q,
                      # 'predicted_disttype': p_hat,
                      'predicted_quality': q_hat,
                      # 'accuracy': acc,
                      'srcc': srcc,
                      'plcc': plcc}
            torch.save(result, save_file)

        return srcc, plcc
    """

    def _evaluateImage_denseCrop(self, test_config):
        if test_config is None:
            return None, None
# LQ add it
#         self.test_data = ImageQualityDataset(csv_file=test_config['input_csv'], root_dir=test_config['root_dir'], enable_dist=self.enable_dist_test, transform=self.test_transform(128))
      #  self.test_data = ImageQualityDatasetLQFull(csv_file_dist=test_config['input_csv'], csv_file_mos=test_config['input_csv'], root_dir_dist=test_config['root_dir'], root_dir_mos=test_config['root_dir'], enable_dist=self.enable_dist_test, transform=self.test_transform(128))
      #  print(self.test_data[0]["image"].shape)
        self.test_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=test_config['input_csv'], csv_file_mos=test_config['input_csv'], root_dir_dist=test_config['root_dir'], root_dir_mos=test_config['root_dir'], enable_dist=self.enable_dist_test, transform=self.test_transform(128))
       # self.train_data = ImageQualityDatasetLQSixSeperate(csv_file_dist=config.train_csv, csv_file_mos=config.train_csv_DT, root_dir_dist=config.trainsetDT, root_dir_mos=config.trainset, enable_dist=self.enable_dist_test, transform=self.test_transform, train=config.train)
       # print(self.test_data[0]["image"].shape)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=test_config['test_batch_size'],
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=test_config['num_workers'],
                                      collate_fn=self.patch_collate)

        length = len(self.test_loader.dataset)
        image_name_list = []
        score_predict_list = np.zeros([length])
        score_list = np.zeros([length])
        if self.enable_dist_test:
            disttype_list = np.zeros([length], dtype=np.int)
            disttype_predict_list = np.zeros([length, self.output_channel])
        batch_size = test_config['test_batch_size']
        # score_predict_noPristine_list = []
        # score_noPristine_list = []
        for counter, sample_batched in enumerate(self.test_loader, 0):

            start_time = time.time()

            if self.enable_dist_test:
                image_batch, score_batch, disttype_batch, name_batch, patch_num_batch = sample_batched['image'], \
                                                                                        sample_batched['score'], \
                                                                                        sample_batched['disttype'], \
                                                                                        sample_batched['image_name'], \
                                                                                        sample_batched['patch_num']
            else:
                image_batch, score_batch, name_batch, patch_num_batch = sample_batched['image'], \
                                                                        sample_batched['score'], \
                                                                        sample_batched['image_name'], \
                                                                        sample_batched['patch_num']
            image = Variable(image_batch)  # shape: (batch_size, channel, H, W)
            #print(image.shape)
            if self.use_cuda:
                image = image.cuda()
            score = score_batch.float().numpy()
            if self.enable_dist_test:
                disttype = disttype_batch.int().numpy()
            #LQ add it for dropout and batch normal operation.
            self.model.eval()
            disttype_predict, score_predict = self.model(image)
            score_predict = score_predict.cpu().data.numpy()                # shape: (batch_size)
            if self.enable_dist_test:
                disttype_predict = disttype_predict.cpu().data.numpy()          # shape: (batch_size, output_channel)

            patch_counter = 0
            for i in range(len(patch_num_batch)):
                score_predict_list[counter * batch_size + i] = np.mean(score_predict[patch_counter: patch_counter + patch_num_batch[i]])   # 1
                # print(score_predict_list[counter * batch_size + i])
                if self.enable_dist_test:
                    disttype_predict_list[counter * batch_size + i] = np.mean(disttype_predict[patch_counter: patch_counter + patch_num_batch[i], :], axis=0)  # 9
                patch_counter += patch_num_batch[i]

            score_list[counter * batch_size: (counter + 1) * batch_size] = score
            if self.enable_dist_test:
                disttype_list[counter * batch_size: (counter + 1) * batch_size] = disttype
            image_name_list += name_batch

            stop_time = time.time()

            samples_per_sec = batch_size / (stop_time - start_time)
            if batch_size == 1:
                print(counter, "/", length, name_batch[0], score[0], score_predict[0], score[0] - score_predict[0], '\tSamples/Sec', samples_per_sec)
            else:
                print(batch_size, counter * batch_size, "/", length, '\tSamples/Sec', samples_per_sec)

        srcc = scipy.stats.mstats.spearmanr(x=score_list, y=score_predict_list)[0]
        plcc = scipy.stats.mstats.pearsonr(x=score_list, y=score_predict_list)[0]

        if self.enable_dist_test:
            max_idxs = np.argmax(disttype_predict_list, axis=1)
            acc = np.sum(np.equal(max_idxs, disttype_list)) / len(disttype_list)

        if self.train:
            self.writer.add_scalar('Test/TestSRCC', srcc, self.current_epoch * self.num_steps_per_epoch)
            self.writer.add_scalar('Test/TestPLCC', plcc, self.current_epoch * self.num_steps_per_epoch)
            if self.enable_dist_test:
                self.writer.add_scalar('Test/TestAcc', acc, self.current_epoch * self.num_steps_per_epoch)

        if self.enable_dist_test:
            print("SRCC: ", srcc, "PLCC: ", plcc, "Acc: ", acc)
        else:
            print("SRCC: ", srcc, "PLCC: ", plcc)

        if test_config['save_path'] is not None:
            save_path = os.path.join(test_config['save_path'], self.model_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, self.model_name + '_' + str(self.current_epoch) + '_' + test_config['name'] + '.pt')
            if self.enable_dist_test:
                result = {'db_name': test_config['name'],
                          'model_name': self.model_name,
                          'image_names': image_name_list,
                          'gt_quality': score_list,
                          'predicted_quality': score_predict_list,
                          'srcc': srcc,
                          'plcc': plcc,
                          'acc': acc,
                          }
            else:
                result = {'db_name': test_config['name'],
                          'model_name': self.model_name,
                          'image_names': image_name_list,
                          'gt_quality': score_list,
                          'predicted_quality': score_predict_list,
                          'srcc': srcc,
                          'plcc': plcc,
                          }
            # torch.save(result, save_file)

        test_result_file = os.path.join(save_path, self.model_name + '_' + str(self.current_epoch) + '_' + test_config['name'] + '.txt')

        if self.enable_dist_test:
            np.savetxt(test_result_file, np.column_stack([image_name_list, score_predict_list, score_list, max_idxs, disttype_list]), fmt="%s", delimiter=",")
            return srcc, plcc, acc
        else:
            np.savetxt(test_result_file, np.column_stack([image_name_list, score_predict_list, score_list]), fmt="%s", delimiter=",")
            return srcc, plcc

    def _train_single_epoch(self, epoch):
        # initialize logging system
        self.current_epoch = epoch
        # num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * self.num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training

        print('Adam learning rate: {:e}'.format(self.optimizer.param_groups[0]['lr']))
     #   print('SGD learning rate: {:e}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):

            images_batch, score_batch, disttype_batch= sample_batched['image'], sample_batched['score'], sample_batched['disttype']

            image = Variable(images_batch)  # shape: (batch_size, channel, H, W)
            score = Variable(score_batch.float())  # shape: (batch_size)
            disttype = Variable(disttype_batch.long())  # shape: (batch_size)

            if self.use_cuda:
                score = score.cuda(async=True)
                disttype = disttype.cuda(async=True)
                image = image.cuda(async=True)

            self.optimizer.zero_grad()
            y, q = self.model(image)

            batch_size = int(q.nelement() / 1)
            y_avg = y.view(batch_size, 1, -1).mean(1)  # shape: (batch_size, output_channel)
            q_avg = q.view(batch_size, 1).mean(1)  # shape: (batch_size)

            # print("y_avg shape", str(y_avg.shape))
            # print("disttype shape", str(disttype.shape))

            _, max_idxs = y_avg.max(dim=1)
            train_acc = np.sum(np.equal(max_idxs.data.cpu().numpy(), disttype_batch.numpy())) / len(disttype_batch)
            # print('Training accuracy:', train_acc)

            self.loss_dt = self.crit_dist_type(y_avg, disttype)
            self.loss_qp = self.crit_quality_pred(q_avg, score)
            self.loss = self.loss_dt + self.lamb * self.loss_qp ##LQ: the loss function
           # self.loss = -self.loss_qp
            self.loss.backward()
            self.optimizer.step()

            self._gdn_param_proc()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)
            if self.train:
                self.writer.add_scalar('Train/TrainLossDT', self.loss_dt.item(), local_counter)
                self.writer.add_scalar('Train/TrainLossQP', self.loss_qp.item(), local_counter)
                self.writer.add_scalar('Train/TrainLoss', self.loss.item(), local_counter)
                self.writer.add_scalar('Train/TrainAcc', train_acc, local_counter)

            lr = self.optimizer.param_groups[0]['lr']
            if self.train:
                self.writer.add_scalar('lr', lr, local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = '(E:%d, S:%d) [loss_qp = %.4f, loss_dt = %.4f, total loss = %.4f, acc = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
            print_str = format_str % (epoch, step, self.loss_qp.item(), self.loss_dt.item(), self.loss.item(), train_acc, lr, examples_per_sec, duration_corrected)
            print(print_str)

            local_counter += 1
            start_time = time.time()

            self.train_loss.append(loss_corrected)
            self.train_dt_loss.append(self.loss_dt.item())
            self.train_qp_loss.append(self.loss_qp.item())

            if self.lr_scheduler_name == "CosineAnnealingLR":
                self.scheduler.step()

        if self.lr_scheduler_name == "StepLR":
            self.scheduler.step()

        if (epoch + 1) % self.every_eval == 0:
            # evaluate after every other epoch
            val_results = self.eval_test(self.val_config)
            out_str = 'Epoch {}'.format(epoch)
            out_str += ' Validating '
            for db_name in val_results:
                if db_name in self.val_results:
                    self.val_results[db_name].append(val_results[db_name])
                else:
                    self.val_results[db_name] = [val_results[db_name]]
                result = val_results[db_name]
                if self.enable_dist_test:
                    out_str += '\n' + db_name + ' srcc: ' + str(result[0]) + '\tplcc: ' + str(result[1]) + '\tAcc: ' + str(result[2])
                else:
                    out_str += '\n' + db_name + ' srcc: ' + str(result[0]) + '\tplcc: ' + str(result[1])
            print(out_str)
            self.val_srcc.append(result[0])
            self.val_plcc.append(result[1])
            if self.enable_dist_test:
                self.val_acc.append(result[2])

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            if not os.path.isdir(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                'train_loss': self.train_loss,
                'train_dt_loss': self.train_dt_loss,
                'train_qp_loss': self.train_qp_loss,
                'val_srcc': self.val_srcc,
                'val_plcc': self.val_plcc,
                'val_acc': self.val_acc,
                'val_results': self.val_results,
                'train_batch_per_epoch': math.ceil(self.train_data_size / self.train_batch_size),
                'every_eval': self.every_eval,
            }, model_name)



    def eval_test(self, *args):
        """
        Evaluate distortion type accuracy and quality prediction SRCC for all test databases in args.
        All results are saved in self.test_results.
        :param args: dicts of configurations for evaluating databases
        :return results: a dictionary containing classification accuracies and SRCCs for all eval databases
        """

        results = {}
        for val_config in args:
            val_config = self._complete_config(val_config)
            db_name = val_config["name"]
            print('Evaluating: {} database'.format(db_name))
            if self.enable_dist_test:
                val_srcc, val_plcc, acc = self._evaluateImage_denseCrop(val_config)
                results[db_name] = [val_srcc, val_plcc, acc]
            else:
                print("==== No dist type in testing this dataset ====")
                val_srcc, val_plcc = self._evaluateImage_denseCrop(val_config)
                results[db_name] = [val_srcc, val_plcc]
        return results

    # gdn parameter post-processing
    def _gdn_param_proc(self):
        for m in self.model.modules():
            if isinstance(m, Gdn):
                m.beta.data.clamp_(min=2e-10)
                m.gamma.data.clamp_(min=2e-10)
                m.gamma.data = (m.gamma.data + m.gamma.data.t()) / 2

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            # start_epoch = checkpoint['epoch'] + 1
            self.current_epoch = checkpoint['epoch']
            self.train_loss = checkpoint['train_loss']
            self.train_dt_loss = checkpoint['train_dt_loss']
            self.train_qp_loss = checkpoint['train_qp_loss']
            self.val_srcc = checkpoint['val_srcc']
            self.val_plcc = checkpoint['val_plcc']
            self.val_acc = checkpoint['val_acc']
            self.val_results = checkpoint['val_results']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # if self.initial_lr is not None:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            # print("[!] no checkpoint found at '{}'".format(ckpt))
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))

    def _load_DT_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            pretrain_model_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()
            for name, param in pretrain_model_dict.items():
                model_dict[name].copy_(param)

            print("[*] loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))
        else:
            # print("[!] no checkpoint found at '{}'".format(ckpt))
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)


