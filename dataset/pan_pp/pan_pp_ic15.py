import math
import random
import string

import cv2
import mmcv
import numpy as np
import Polygon as plg
import pyclipper
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

ic15_root_dir = './data/ICDAR2015/Challenge4/'
ic15_train_data_dir = ic15_root_dir + 'ch4_training_images/'
ic15_train_gt_dir = ic15_root_dir + \
                    'ch4_training_localization_transcription_gt/'
ic15_test_data_dir = ic15_root_dir + 'ch4_test_images/'
ic15_test_gt_dir = ic15_root_dir + \
                   'ch4_test_localization_transcription_gt/'



vin_root_dir = './data/ICDAR2015/vietnamese_orgin/'

vin_train_data_dir  = vin_root_dir +    'train_images/'
vin_train_gt_dir    = vin_root_dir +    'labels/'
vin_test_data_dir   = vin_root_dir +    'test_images/'
vin_test_gt_dir     = vin_root_dir +    'labels/'


dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D^", "d^"]


def parse_tone(word):
    res = ""
    tone = ""
    for char in word:
        if char in dictionary:
            for group in groups:
                if char in group:
                    if tone == "":
                        tone = TONES[group.index(char)]
                    res += group[0]
        else:
            res += char
    res += tone
    return res


def full_parse(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res


def correct_tone_position(word):
    word = word[:-1]
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    replace_char = correct_tone_position(recognition)
    if recognition[-1] in TONES:
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition


#print(decoder(full_parse("Rồng")))


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception:
        print('Cannot read image: %s.' % img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word= gt[8] if len(gt) ==9 else ','.join(gt[8:])
        word = word.replace('\r', '').replace('\n', '')
        word = full_parse(word)
        if len(word)==0:
            word ="###"
        if word[0] == '#':
            words.append('###')
        else:
            words.append(word)

        bbox = [int(gt[i]) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

#Augument
def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img,
                                      rotation_matrix, (h, w),
                                      flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs

#Augument
def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

#Augument
def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

#Augument
def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)

    img = scale_aligned(img, h_scale, w_scale)
    return img

#Augument
def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0
                                                   for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img,
                                       0,
                                       p_h - t_h,
                                       0,
                                       p_w - t_w,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0, ))
        n_imgs.append(img_p)
    return n_imgs


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5),
                         max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox)[0]
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception:
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    elif voc_type =='VIETCASE':
        tones =[' ','ˋ', 'ˊ', '﹒', 'ˀ', '˜', 'ˇ', 'ˆ', '˒', '‑']

        voc = list(string.printable[:-6])
        voc+= tones 
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", '
                       '"ALLCASES", "ALLCASES_SYMBOLS" ,"VIETCASE"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


class PAN_PP_IC15(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_scale=0.5,
                 with_rec=False,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (
            img_size is None or isinstance(img_size, tuple)) else (img_size,
                                                                   img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type

        if split == 'train':
            data_dirs = [ic15_train_data_dir, vin_train_data_dir]
            gt_dirs = [ic15_train_gt_dir, vin_train_gt_dir]
        elif split == 'test':
            data_dirs = [ic15_test_data_dir, vin_test_data_dir]
            gt_dirs = [ic15_test_gt_dir, vin_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        count=0
        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [
                img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')
            ]
            img_names.extend([
                img_name for img_name in mmcv.utils.scandir(data_dir, '.png')
            ])

            img_names.extend([
                img_name for img_name in mmcv.utils.scandir(data_dir, '.jpge')
            ])

            img_paths = []
            gt_paths = [] 
            if count ==0:
                for idx, img_name in enumerate(img_names): 
                    
                    img_path = data_dir + img_name
                    img_paths.append(img_path)

                    gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                    gt_path = gt_dir + gt_name
                    gt_paths.append(gt_path)

                self.img_paths.extend(img_paths)
                self.gt_paths.extend(gt_paths)
                print("Found {} in folder {}".format(len(img_paths), count)
            else:
                for idx, img_name in enumerate(img_names): 
                    
                    img_path = data_dir + img_name
                    img_paths.append(img_path)

                    gt_name = img_name.split('.')[0][2:] 
                    gt_name= str(int(gt_name))
                    gt_name = 'gt_' + gt_name + '.txt' 
                    gt_path = gt_dir + gt_name
                    gt_paths.append(gt_path)

                self.img_paths.extend(img_paths)
                self.gt_paths.extend(gt_paths)
                print("Found {} in folder {}".format(len(img_paths), count)

            count+=1

        print("Found total {} image and ground truth in {} set!".format(len(self.img_paths), split))

        if report_speed:
            target_size = 3000
            extend_scale = (target_size + len(self.img_paths) - 1) // len(
                self.img_paths)
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.voc, self.char2id, self.id2char = get_vocabulary('VIETCASE')
        self.max_word_num = 200
        self.max_word_len = 32
        print('reading type: %s.' % self.read_type)

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):

        #init path
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        #read image, bboxes, annotation
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)

        #fit count box == count word
        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        #embding char to int
        #   setup
        gt_words = np.full((self.max_word_num + 1, self.max_word_len),
                           self.char2id['PAD'],
                           dtype=np.int32)
        word_mask = np.zeros((self.max_word_num + 1, ), dtype=np.int32)

        #   convert
        for i, word in enumerate(words):
            if word == '###':
                continue
            #word = word.lower()
            gt_word = np.full((self.max_word_len, ),
                              self.char2id['PAD'],
                              dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1
            

        #transform? Why do it transform here? It can make mistake bboxes
        if self.is_transform:
            #pass
            img = random_scale(img, self.short_size)

        
        #draw gt_mask with 4 corner (if ### draw into traning_mask (for detection))
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        #init kernels from bboxes
        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        #transform
        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
            #update word and mask because transformation can lose box
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop,
                                         word_mask)
        
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        #get bounding box after transformation
        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255,
                                         saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )
        if self.with_rec:
            data.update(dict(gt_words=gt_words, word_masks=word_mask))

        return data

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(org_img_size=np.array(img.shape[:2]))

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2])))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        data = dict(imgs=img, img_metas=img_meta)

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)


# input --image
#       --bboxes
#       --instance mask
#       --annotation

# out   --Image
#       --bboxes 
#       --text_mask (for dectection)
#       --instance_mask (for recognition)
#       --kernels_mask (like anchor or bezier method)
#       --gt_word (after embding)
#       --word_mask (?)