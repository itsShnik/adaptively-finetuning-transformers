import os
import json
import _pickle as cPickle
from PIL import Image
import re
import base64
import numpy as np
import csv
import sys
import time
import pprint
import logging

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

from pycocotools.coco import COCO

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


class VQA_CP(Dataset):
    def __init__(self, image_set, root_path, data_path, answer_vocab_file, use_imdb=True,
                 with_precomputed_visual_feat=False, boxes="36",
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=True, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, toy_dataset=False, toy_samples=128, **kwargs):
        """
        Visual Question Answering Dataset

        :param image_set: image folder name
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(VQA_CP, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'

        categories = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                      'boat',
                      'trafficlight', 'firehydrant', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse',
                      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                      'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove',
                      'skateboard', 'surfboard', 'tennisracket', 'bottle', 'wineglass', 'cup', 'fork', 'knife', 'spoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut',
                      'cake', 'chair', 'couch', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tv', 'laptop', 'mouse',
                      'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                      'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush']
        vqa_question = {
            "train": "vqa/vqacp_v2_train_questions.json",
            "val": "vqa/vqacp_v2_test_questions.json",
        }
        vqa_annot = {
            "train": "vqa/vqacp_v2_train_annotations.json",
            "val": "vqa/vqacp_v2_test_annotations.json",
        }
        
        if boxes == "36":
            precomputed_boxes = {
                'train': ("vgbua_res101_precomputed", "{}_resnet101_faster_rcnn_genome_36"),
                'val': ("vgbua_res101_precomputed", "{}_resnet101_faster_rcnn_genome_36"),
            }
        elif boxes == "10-100ada":
            precomputed_boxes = {
                'train': ("vgbua_res101_precomputed", "{}_resnet101_faster_rcnn_genome"),
                'val': ("vgbua_res101_precomputed", "{}_resnet101_faster_rcnn_genome"),
            }
        else:
            raise ValueError("Not support boxes: {}!".format(boxes))

        self.coco_dataset = {
            "train2014": os.path.join(data_path, "annotations", "instances_train2014.json"),
            "val2014": os.path.join(data_path, "annotations", "instances_val2014.json"),
            "test-dev2015": os.path.join(data_path, "annotations", "image_info_test-dev2015.json"),
            "test2015": os.path.join(data_path, "annotations", "image_info_test2015.json"),
        }

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

        self.boxes = boxes
        self.test_mode = test_mode
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.category_to_idx = {c: i for i, c in enumerate(categories)}
        self.data_path = data_path
        self.root_path = root_path

        # load the answer vocab file: same as vqav2 dataset
        with open(answer_vocab_file, 'r', encoding='utf8') as f:
            self.answer_vocab = [w.lower().strip().strip('\r').strip('\n').strip('\r') for w in f.readlines()]
            self.answer_vocab = list(filter(lambda x: x != '', self.answer_vocab))
            self.answer_vocab = [self.processPunctuation(w) for w in self.answer_vocab]

        # The config.DATA.TRAIN_IMAGE_SET and config.DATA.VAL_IMAGE_SET have
        # a little different use here, it indicates the mode 'train' or 'val'
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.ann_files = [os.path.join(data_path, vqa_annot[iset]) for iset in self.image_sets] \
            if not self.test_mode else [None for iset in self.image_sets]
        self.q_files = [os.path.join(data_path, vqa_question[iset]) for iset in self.image_sets]

        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset][0], precomputed_boxes[iset][1]) for iset in self.image_sets]

        self.box_bank = {}
        self.coco_datasets = [os.path.join(data_path, '{}', 'COCO_{}_{{:012d}}.jpg') for iset in self.image_sets]

        self.transform = transform
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        self.mask_size = mask_size

        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        if zip_mode:
            self.zipreader = ZipReader()

        self.database = self.load_annotations()
        if self.aspect_grouping:
            self.group_ids = self.group_aspect(self.database)

        # toy dataset
        if toy_dataset:
            print(f"Using the toy dataset!! Total samples = {toy_samples}")
            self.database = self.database[:toy_samples]

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'question']
        else:
            return ['image', 'boxes', 'im_info', 'question', 'label']

    def __getitem__(self, index):
        idb = self.database[index]

        # image, boxes, im_info
        boxes_data = self._load_json(idb['box_fn'])
        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = idb['width'], idb['height']

            boxes_features = torch.as_tensor(
                np.frombuffer(self.b64_decode(boxes_data['features']), dtype=np.float32).reshape((boxes_data['num_boxes'], -1))
            )
        else:
            image = self._load_image(idb['image_fn'])
            w0, h0 = image.size
        boxes = torch.as_tensor(
            np.frombuffer(self.b64_decode(boxes_data['boxes']), dtype=np.float32).reshape(
                (boxes_data['num_boxes'], -1))
        )

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                if 'image_box_feature' in boxes_data:
                    image_box_feature = torch.as_tensor(
                        np.frombuffer(
                            self.b64_decode(boxes_data['image_box_feature']), dtype=np.float32
                        ).reshape((1, -1))
                    )
                else:
                    image_box_feature = boxes_features.mean(0, keepdim=True)
                boxes_features = torch.cat((image_box_feature, boxes_features), dim=0)
        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # flip: 'left' -> 'right', 'right' -> 'left'
        q_tokens = self.tokenizer.tokenize(idb['question'])
        if flipped:
            q_tokens = self.flip_tokens(q_tokens, verbose=False)
        if not self.test_mode:
            answers = idb['answers']
            if flipped:
                answers_tokens = [a.split(' ') for a in answers]
                answers_tokens = [self.flip_tokens(a_toks, verbose=False) for a_toks in answers_tokens]
                answers = [' '.join(a_toks) for a_toks in answers_tokens]
            label = self.get_soft_target(answers)

        # question
        q_retokens = q_tokens
        q_ids = self.tokenizer.convert_tokens_to_ids(q_retokens)

        # concat box feature to box
        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=-1)

        if self.test_mode:
            return image, boxes, im_info, q_ids
        else:
            # print([(self.answer_vocab[i], p.item()) for i, p in enumerate(label) if p.item() != 0])
            return image, boxes, im_info, q_ids, label

    @staticmethod
    def flip_tokens(tokens, verbose=True):
        changed = False
        tokens_new = [tok for tok in tokens]
        for i, tok in enumerate(tokens):
            if tok == 'left':
                tokens_new[i] = 'right'
                changed = True
            elif tok == 'right':
                tokens_new[i] = 'left'
                changed = True
        if verbose and changed:
            logging.info('[Tokens Flip] {} -> {}'.format(tokens, tokens_new))
        return tokens_new

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def answer_to_ind(self, answer):
        if answer in self.answer_vocab:
            return self.answer_vocab.index(answer)
        else:
            return self.answer_vocab.index('<unk>')

    def get_soft_target(self, answers):

        soft_target = torch.zeros(len(self.answer_vocab), dtype=torch.float)
        answer_indices = [self.answer_to_ind(answer) for answer in answers]
        gt_answers = list(enumerate(answer_indices))
        unique_answers = set(answer_indices)

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]

                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)

            if answer != self.answer_vocab.index('<unk>'):
                soft_target[answer] = avg_acc

        return soft_target

    def processPunctuation(self, inText):

        if inText == '<unk>':
            return inText

        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def load_annotations(self):
        tic = time.time()
        database = []
        db_cache_name = 'vqa_cp2_boxes{}_{}'.format(self.boxes, '+'.join(self.image_sets))
        if self.with_precomputed_visual_feat:
            db_cache_name += 'visualprecomp'
        if self.zip_mode:
            db_cache_name = db_cache_name + '_zipmode'
        if self.test_mode:
            db_cache_name = db_cache_name + '_testmode'
        db_cache_root = os.path.join(self.root_path, 'cache')
        db_cache_path = os.path.join(db_cache_root, '{}.pkl'.format(db_cache_name))

        if os.path.exists(db_cache_path):
            if not self.ignore_db_cache:
                # reading cached database
                print('cached database found in {}.'.format(db_cache_path))
                with open(db_cache_path, 'rb') as f:
                    print('loading cached database from {}...'.format(db_cache_path))
                    tic = time.time()
                    database = cPickle.load(f)
                    print('Done (t={:.2f}s)'.format(time.time() - tic))
                    return database
            else:
                print('cached database ignored.')

        # ignore or not find cached database, reload it from annotation file
        print('loading database of split {}...'.format('+'.join(self.image_sets)))
        tic = time.time()

        for ann_file, q_file, coco_path, box_file \
                in zip(self.ann_files, self.q_files, self.coco_datasets, self.precomputed_box_files):
            qs = self._load_json(q_file)
            anns = self._load_json(ann_file) if not self.test_mode else ([None] * len(qs))

            # we need to create 3 coco objects
            coco_train2014 = COCO(self.coco_dataset['train2014'])
            coco_val2014 = COCO(self.coco_dataset['val2014'])
            coco_test2015 = COCO(self.coco_dataset['test2015'])
            for ann, q in zip(anns, qs):
                if q['coco_split'] == 'train2014':
                    coco_obj = coco_train2014
                    box_dir = 'trainval2014'
                elif q['coco_split'] == 'val2014':
                    coco_obj = coco_val2014
                    box_dir = 'trainval2014'
                elif q['coco_split'] == 'test2015':
                    coco_obj = coco_test2015
                    box_dir = 'test2015'
                else:
                    raise ValueError("COCO split in question : {} not supported".format(q['coco_split']))

                idb = {'image_id': q['image_id'],
                       'image_fn': coco_path.format(q['coco_split'], q['coco_split'], q['image_id']),
                       'width': coco_obj.imgs[q['image_id']]['width'],
                       'height': coco_obj.imgs[q['image_id']]['height'],
                       'box_fn': os.path.join(box_file.format(box_dir), '{}.json'.format(q['image_id'])),
                       'question_id': q['question_id'],
                       'question': q['question'],
                       'answers': [a['answer'] for a in ann['answers']] if not self.test_mode else None,
                       'multiple_choice_answer': ann['multiple_choice_answer'] if not self.test_mode else None,
                       "question_type": ann['question_type'] if not self.test_mode else None,
                       "answer_type": ann['answer_type'] if not self.test_mode else None,
                       }
                database.append(idb)

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        # cache database via cPickle
        if self.cache_db:
            print('caching database to {}...'.format(db_cache_path))
            tic = time.time()
            if not os.path.exists(db_cache_root):
                makedirsExist(db_cache_root)
            with open(db_cache_path, 'wb') as f:
                cPickle.dump(database, f)
            print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def load_precomputed_boxes(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = {}
            with open(box_file, "r") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in (['boxes', 'features'] if self.with_precomputed_visual_feat else ['boxes']):
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                                    dtype=np.float32).reshape((item['num_boxes'], -1))
                    in_data[item['image_id']] = item
            self.box_bank[box_file] = in_data
            return in_data

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

