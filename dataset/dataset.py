# -*- coding: utf-8 -*-

import numpy as np
import torch as t
from PIL import Image
from torchvision import transforms as T
import cv2
from matplotlib import pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# training set 的 mean 和 std
# >>> train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
# >>> l = [x[0] for x in tqdm(train_data)]
# >>> x = t.cat(l, 0)
# >>> x.mean()
# >>> x.std()
MURA_MEAN = [0.22588661454502146] * 3
MURA_STD = [0.17956269377916526] * 3


def logo_filter(data, threshold=200):

    im = Image.new('L', data.size)

    list_data = list(data.split()[0].getdata())

    pixels = [x if x < threshold else 0 for x in list_data]

    im.putdata(data=pixels)

    return im


def crop_minAreaRect(img, rect):
    size = rect[1]
    angle = rect[2]
    (o_h, o_w) = size[0], size[1]
    (cX, cY) = rect[0]

    rotated = False
    angle = rect[2]

    # if (angle < -45 and (h > w)) or angle == -90 :
    if (angle < -45) or angle == -90:
        angle += 90
        rotated = True
    w = o_h if rotated == False else o_w
    h = o_w if rotated == False else o_h

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    img_rot = cv2.warpAffine(img, M, (nW, nH))
    img_crop = cv2.getRectSubPix(img_rot, (int(w * 0.9), int(h * 0.9)), (nW / 2, nH / 2))

    if abs(rect[2]) < 3.0:
        img_crop = cv2.getRectSubPix(img, (int(w), int(h)), (cX, cY))

    if img_crop[:, :, 2].mean() > 170:  # or angle == 0.0:
        img_crop = img

    return img_crop
    #
    # size = rect[1]
    # angle = rect[2]
    # (h, w) = size[0], size[1]
    # (cX, cY) = rect[0]
    # rotated = False
    # angle = rect[2]
    #
    # if (angle < -45 and (h > w)) or angle == -90 :
    #     angle += 90
    #     rotated = True
    #
    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    # # compute the new bounding dimensions of the image
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY
    # # perform the actual rotation and return the image
    # img_rot = cv2.warpAffine(img, M, (nW, nH))
    #
    # r_w = h if rotated==False else w
    # r_h = w if rotated==False else h
    # img_crop = cv2.getRectSubPix(img_rot, (int(r_w * 0.9), int(r_h * 0.9)), (nW / 2, nH / 2))
    # if abs(rect[2]) < 3.0:
    #     img_crop = cv2.getRectSubPix(img, (int(h), int(w)), (cX, cY))
    #
    #
    # if img_crop[:, :, 2].mean() > 170: #or angle == 0.0:
    #     img_crop = img
    # return img_crop


def align_mura_elbow(img):
    data = Image.fromarray(img)
    img2 = logo_filter(data)
    # opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
    imgray = np.array(img2)

    # qqqqqqqqqimgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray, (7, 7), 4)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x = 750.0
    removed_contour = []
    for c in contours:
        # print(cv2.contourArea(c))
        if cv2.contourArea(c) > x:
            # print(cv2.contourArea(c))
            removed_contour.append(c)
    if len(removed_contour) == 0:
        removed_contour = contours
    cnt = np.concatenate(removed_contour)
    rotrect = cv2.minAreaRect(cnt)

    # box = np.int0(cv2.boxPoints(rotrect))

    # crop
    img_croped = crop_minAreaRect(img, rotrect)
    return img_croped


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class MURA_Dataset(object):

    def __init__(self, root, csv_path, part='all', transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        train set:     train = True,  test = False
        val set:       train = False, test = False
        test set:      train = False, test = True

        part = 'all', 'XR_HAND', XR_ELBOW etc.
        用于提取特定部位的数据。
        """

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if part == 'all':
                imgs = [root + str(x, encoding='utf-8').strip() for x in d]  # 所有图片的存储路径, [:-1]目的是抛弃最末尾的\n
            else:
                imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                        str(x, encoding='utf-8').strip().split('/')[2] == part]

        self.imgs = imgs
        self.train = train
        self.test = test
        self.max_width = 0
        self.max_height = 0

        if transforms is None:

            if self.train and not self.test:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Lambda(logo_filter),
                    T.Resize(320),
                    T.RandomCrop(320),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(30),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            if not self.train:
                # 这里的X光图是1 channel的灰度图
                self.transforms = T.Compose([
                    T.Lambda(logo_filter),
                    T.Resize(320),
                    T.CenterCrop(320),
                    T.ToTensor(),
                    T.Lambda(lambda x: t.cat([x[0].unsqueeze(0), x[0].unsqueeze(0), x[0].unsqueeze(0)], 0)),  # 转换成3 channel
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据：data, label, path, body_part
        """

        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)

        ## align
        # if self.train and not self.test:
        #     img_path = self.imgs[index]
        #     img = cv2.imread(img_path)
        #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img_croped = align_mura_elbow(img)
        #     self.max_height = img.shape[0] if img.shape[0] > self.max_height else self.max_height
        #     self.max_width = img.shape[1] if img.shape[1] > self.max_width else self.max_width
        #     img_croped = cv2.cvtColor(img_croped, cv2.COLOR_BGR2GRAY)
        #     # contrast limit가 2이고 title의 size는 8X8
        #     clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4, 4))
        #     img_croped = clahe.apply(img_croped)
        #     img = img_croped
        #
        # if not self.train:
        #     img_path = self.imgs[index]
        #     img = cv2.imread(img_path)
        #
        # data = Image.fromarray(img)
        # # data = Image.open(img_path)
        # data = self.transforms(data)


        # label
        if not self.test:
            label_str = img_path.split('_')[-1].split('/')[0]
            if label_str == 'positive':
                label = 1
            elif label_str == 'negative':
                label = 0
            else:
                print(img_path)
                print(label_str)
                raise IndexError

        if self.test:
            label = 0

        # body part
        body_part = img_path.split('/')[6]

        return data, label, img_path, body_part

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # from config.config import opt
    # from tqdm import tqdm
    # train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, train=True)
    # l = [x[0] for x in tqdm(train_data)]
    # x = t.cat(l, 0)
    # print(x.mean())
    # print(x.std())
    from tqdm import tqdm
    from config import opt

    show_image_hist = True
    #data_root = '/home/data/extra/'

    # train_labeled_studies 和 test_labeled_studies 不需要，根据folder的名字来判断label
    #train_image_paths = data_root + 'MURA-v1.1/train_image_paths.csv'   # 训练集存放路径
    part = opt.part
    train_data = MURA_Dataset(opt.data_root, opt.train_image_paths, part=part, train=True)

    if show_image_hist:
        l = [(x[0].cpu().numpy(), x[2]) for x in tqdm(train_data)]
    else:
        l = np.array([x[0].cpu().numpy() for x in tqdm(train_data)])
    # m = t.cat(l, 0)

    fig, axes = plt.subplots(nrows=2, ncols=2)

    if show_image_hist:
        for data, file_path in l:
            fig.suptitle(str(file_path))
            np_array = data
            np_array2 = np_array.transpose(2, 1, 0)
            np_array2 = np.asarray(np_array2 * 255, dtype='uint8')
            axes[0, 0].imshow(np_array2)
            hist_full = np.histogram(np_array2[:, :, 0], bins=256, range=[0, 256])

            count_f = hist_full[0]
            bins = hist_full[1]
            axes[0, 1].bar(bins[:-1], count_f, color='r', alpha=0.5)

            img_croped = cv2.cvtColor(np_array.transpose(2, 1, 0), cv2.COLOR_BGR2GRAY)
            img_croped = np.asarray(img_croped * 255, dtype='uint8')

            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4, 4))
            img2 = clahe.apply(img_croped)
            axes[1, 0].imshow(img2, cmap='gray')
            hist_full_ep = np.histogram(img2, bins=256, range=[0, 256])

            count_f_eq = hist_full_ep[0]
            bins_eq = hist_full_ep[1]
            axes[1, 1].bar(bins_eq[:-1], count_f_eq, color='r', alpha=0.5)

            fig.show()
            plt.pause(2)
            plt.clf()
            fig, axes = plt.subplots(nrows=2, ncols=2)
    else:
        print("none")
        print(l.mean())
        print(l.std())

        # print(x.mean())
        # print(x.std())



