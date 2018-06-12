# -*- coding: utf-8 -*-

import torch as t
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# training set 的 mean 和 std
# >>> train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
# >>> l = [x[0] for x in tqdm(train_data)]
# >>> x = t.cat(l, 0)
# >>> x.mean()
# >>> x.std()
MURA_MEAN = [0.22588661454502146]
MURA_STD = [0.17956269377916526]


class MURA_Dataset(object):

    def __init__(self, root, csv_path, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            imgs = [root + str(x, encoding='utf-8')[:-1] for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n

        self.imgs = imgs

        if transforms is None:

            # 这里的X光图是1 channel的灰度图
            self.transforms = T.Compose([
                T.Resize(320),
                T.RandomCrop(320),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.ToTensor(),
                T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, path
        """

        img_path = self.imgs[index]

        label_str = img_path.split('_')[-1].split('/')[0]
        if label_str == 'positive':
            label = 1
        elif label_str == 'negative':
            label = 0
        else:
            raise IndexError

        data = Image.open(img_path)

        data = self.transforms(data)

        return data, label, img_path

    def __len__(self):
        return len(self.imgs)
