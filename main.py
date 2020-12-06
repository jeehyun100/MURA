# -*- coding: utf-8 -*-

import os
import sys
import csv
import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score#, confusion_matrix
import time
import torch.nn as nn
import models
from config import opt
from utils import Visualizer, FocalLoss
from dataset import MURA_Dataset
import cv2


def train(**kwargs):

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    opt.parse(kwargs)
    if opt.use_visdom:
        vis = Visualizer(opt.env)

    # step 1: configure model
    # model = densenet169(pretrained=True)
    # model = DenseNet169(num_classes=2)
    # model = ResNet152(num_classes=2)
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        print("Load model : {0}".format(opt.load_model_path))
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = nn.DataParallel(model)
        print('CUDA MODEL!')
        model.cuda()

    model.train()

    # step 2: data
    # train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, 'XR_ELBOW', train=True, test=False)
    # val_data = MURA_Dataset(opt.data_root, opt.test_image_paths,'XR_ELBOW', train=False, test=False)
    part = opt.part
    train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, part, train=True, test=False)
    val_data = MURA_Dataset(opt.data_root, opt.test_image_paths,part, train=False, test=False)

    print('Training images:', len(train_data), 'Validation images:', len(val_data))

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step 3: criterion and optimizer
    A = 21935
    N = 14873
    weight = t.Tensor([A / (A + N), N / (A + N)])
    if opt.use_gpu:
        weight = weight.cuda()

    criterion = t.nn.CrossEntropyLoss(weight=weight)
    # criterion = FocalLoss(alpha=weight, class_num=2)
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step 4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # step 5: train     # step 5: train
    chk_dir = opt.checkpoint_dir
    #if not os.path.exists(os.path.join('checkpoints', model.model_name)):
    #if not os.path.exists(os.path.join('checkpoints', model.module.model_name)):
    if not os.path.exists(os.path.join(chk_dir, model.module.model_name)):
        #os.mkdir(os.path.join(chk_dir, model.module.model_name), )
        os.makedirs(os.path.join(chk_dir, model.module.model_name), exist_ok=True)
    prefix = time.strftime('%m%d')
    #if not os.path.exists(os.path.join('checkpoints', model.model_name, prefix)):
    if not os.path.exists(os.path.join(chk_dir, model.module.model_name, prefix)):
        #os.mkdir(os.path.join(chk_dir, model.module.model_name, prefix))
        os.makedirs(os.path.join(chk_dir, model.module.model_name, prefix), exist_ok=True)

    s = t.nn.Softmax()
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label, _, body_part) in tqdm(enumerate(train_dataloader)):

            # train model
            input = Variable(data)
            target = Variable(label)
            # body_part = Variable(body_part)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
                # body_part = body_part.cuda()

            optimizer.zero_grad()
            if opt.model.startswith('MultiBranch'):
                score = model(input, body_part)
            else:
                score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            #loss_meter.add(loss.data[0])
            loss_meter.add(loss.cpu().data)
            confusion_matrix.add(s(Variable(score.data)).data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                if opt.use_visdom:
                    vis.plot('loss', loss_meter.value()[0])
                # print('loss', loss_meter.value()[0])

                # debug
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        ck_name = f'epoch_{epoch}_{str(opt)}.pth'
        model.module.save(os.path.join(chk_dir, model.module.model_name, prefix, ck_name))
        # model.save()

        # validate and visualize
        val_cm, val_accuracy, val_loss = val(model, val_dataloader)

        cm = confusion_matrix.value()

        if opt.use_visdom:
            vis.plot('val_accuracy', val_accuracy)
            vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm},train_acc:{train_acc}, "
                     "val_acc:{val_acc}".format(epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
                                         train_cm=str(confusion_matrix.value()), lr=lr,
                                         train_acc=str(100. * (cm[0][0] + cm[1][1]) / (cm.sum())),
                                         val_acc=str(100. * (val_cm.value()[0][0] + val_cm.value()[1][1]) / (val_cm.value().sum()))))
        print('val_accuracy: ', val_accuracy)
        print("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm},train_acc:{train_acc}, "
              "val_acc:{val_acc}".format(epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
                                         train_cm=str(confusion_matrix.value()), lr=lr,
                                         train_acc=100. * (cm[0][0] + cm[1][1]) / (cm.sum()),
                                         val_acc=100. * (val_cm.value()[0][0] + val_cm.value()[1][1]) / (val_cm.value().sum())))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
        # if val_loss > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # previous_loss = val_loss
        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    s = t.nn.Softmax()

    criterion = t.nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    for ii, data in tqdm(enumerate(dataloader)):
        input, label, _, body_part = data
        val_input = Variable(input, volatile=True)
        target = Variable(label)
        # body_part = Variable(body_part)
        if opt.use_gpu:
            val_input = val_input.cuda()
            target = target.cuda()
            # body_part = body_part.cuda()
        if opt.model.startswith('MultiBranch'):
            score = model(val_input, body_part)
        else:
            score = model(val_input)
        # confusion_matrix.add(softmax(score.data.squeeze()), label.type(t.LongTensor))
        if s(Variable(score.data.squeeze())).data.cpu().numpy().shape[0] != label.type(t.LongTensor).cpu().numpy().shape[0]:
            #print("none")
            confusion_matrix.add(s(Variable(score.data)).data, label.type(t.LongTensor))
        else:
            confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))
            #print("same")

        #confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))
        loss = criterion(score, target)
        loss_meter.add(loss.cpu().data)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    loss = loss_meter.value()[0]

    return confusion_matrix, accuracy, loss


def test(**kwargs):
    opt.parse(kwargs)

    # configure model
    # model = DenseNet169(num_classes=2)
    # model = CustomDenseNet169(num_classes=2)
    # model = ResNet152(num_classes=2)
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    model.eval()

    # data
    test_data = MURA_Dataset(opt.data_root, opt.test_image_paths, train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    # confusion_matrix = meter.ConfusionMeter(2)
    # s = t.nn.Softmax()

    for ii, (data, label, path, body_part) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        # body_part = Variable(body_part, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
            # body_part = body_part.cuda()
        if opt.model.startswith('MultiBranch'):
            score = model(input, body_part)
        else:
            score = model(input)

        # confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))

        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()

        # 每一行为 图片路径 和 positive的概率
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results

    # cm_value = confusion_matrix.value()
    # accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    # print('confusion matrix: ')
    # print(cm_value)
    # print(f'accuracy: {accuracy}')

    write_csv(results, opt.result_file)

    calculate_cohen_kappa()
    # return results



def show(**kwargs):
    opt.parse(kwargs)

    # configure model
    # model = DenseNet169(num_classes=2)
    # model = CustomDenseNet169(num_classes=2)
    # model = ResNet152(num_classes=2)
    # model = getattr(models, opt.model)()
    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    # if opt.use_gpu:
    #     model.cuda()
    #
    # model.eval()

    # data
    test_data = MURA_Dataset(opt.data_root, opt.test_image_paths, train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    # confusion_matrix = meter.ConfusionMeter(2)
    # s = t.nn.Softmax()

    for ii, (data, label, path, body_part) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        # body_part = Variable(body_part, volatile=True)
        # if opt.use_gpu:
        #     input = input.cuda()
        #     # body_part = body_part.cuda()
        # if opt.model.startswith('MultiBranch'):
        #     score = model(input, body_part)
        # else:
        #     score = model(input)
        for i in range(data.shape[0]):
            img_data = data.cpu().numpy()[i]
            img_data = np.transpose(img_data, (1, 2, 0))
            # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            cv2.imshow('image', img_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))

        # probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        #
        # # 每一行为 图片路径 和 positive的概率
        # batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        #
        # results += batch_results

    # cm_value = confusion_matrix.value()
    # accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    # print('confusion matrix: ')
    # print(cm_value)
    # print(f'accuracy: {accuracy}')
    #
    # write_csv(results, opt.result_file)
    #
    # calculate_cohen_kappa()
    # return results

def ensemble_test(**kwargs):
    opt.parse(kwargs)

    # configure model
    model_hub = []
    for i in range(len(opt.ensemble_model_types)):
        model = getattr(models, opt.ensemble_model_types[i])()
        if opt.ensemble_model_paths[i]:
            model.load(opt.ensemble_model_paths[i])
        if opt.use_gpu:
            model.cuda()
        model.eval()
        model_hub.append(model)

    # data
    test_data = MURA_Dataset(opt.data_root, opt.test_image_paths, train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    # confusion_matrix = meter.ConfusionMeter(2)
    # s = t.nn.Softmax()

    for ii, (data, label, path) in tqdm(enumerate(test_dataloader)):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()

        probability_hub = []
        for model in model_hub:
            score = model(input)
            # confusion_matrix.add(s(Variable(score.data.squeeze())).data, label.type(t.LongTensor))
            probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
            probability_hub.append(probability)

        # print(probability_hub)

        prob = [np.mean([x[i] for x in probability_hub]) for i in range(len(probability_hub[0]))]

        # 每一行为 图片路径 和 positive的概率
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, prob)]

        results += batch_results

    # cm_value = confusion_matrix.value()
    # accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    # print('confusion matrix: ')
    # print(cm_value)
    # print(f'accuracy: {accuracy}')

    write_csv(results, opt.result_file)

    calculate_cohen_kappa()
    # return results


def write_csv(results, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'probability'])
        writer.writerows(results)


def calculate_cohen_kappa(threshold=0.5):
    input_csv_file_path = 'result.csv'

    result_dict = {}
    with open(input_csv_file_path, 'r') as F:
        d = F.readlines()[1:]
        for data in d:
            (path, prob) = data.split(',')

            folder_path = path[:path.rfind('/')]
            prob = float(prob)

            if folder_path in result_dict.keys():
                result_dict[folder_path].append(prob)
            else:
                result_dict[folder_path] = [prob]

    for k, v in result_dict.items():
        result_dict[k] = np.mean(v)
        # visualize
        # print(k, result_dict[k])

    # 写入每个study的诊断csv
    with open(opt.output_csv_path, 'w') as F:
        writer = csv.writer(F)
        for k, v in result_dict.items():
            path = k[len(opt.data_root):] + '/'
            value = 0 if v >= threshold else 1
            writer.writerow([path, value])

    XR_type_list = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

    for XR_type in XR_type_list:

        # 提取出 XR_type 下的所有folder路径，即 result_dict 中的key
        keys = [k for k, v in result_dict.items() if k.split('/')[6] == XR_type]

        y_true = [1 if key.split('_')[-1] == 'positive' else 0 for key in keys]
        y_pred = [0 if result_dict[key] >= threshold else 1 for key in keys]

        print('--------------------------------------------')

        kappa_score = cohen_kappa_score(y_true, y_pred)

        print(XR_type, kappa_score)

        # 预测准确的个数
        count = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_true))])
        print(XR_type, 'Accuracy', 100.0 * count / len(y_true))


def help(**kwargs):
    """
        打印帮助的信息： python main.py help
    """

    print("""
        usage : python main.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env_MURA' --lr=0.001
                python {0} test --dataset='/path/to/dataset/root/'
                python {0} help
        avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # ------- Train --------
    # import fire
    # fire.Fire()
    train()

    # ------- Test --------
    # opt.test_image_paths = sys.argv[1]
    # opt.output_csv_path = sys.argv[2]
    #test()
    #show()

