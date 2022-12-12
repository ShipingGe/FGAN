import logging
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import numpy as np
import torch
from PIL import Image, ImageFilter
import random
from utils.word2vec import Words2vecs, Doc2vecs
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence

from transformers import BertTokenizer, ViTFeatureExtractor
import gensim

# import clip

logger = logging.getLogger(__name__)

d2v_model = Doc2vecs()

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# device = "cuda" if torch.cuda.is_available() else "cpu"
# _, preprocess = clip.load("ViT-B/32", device="cpu")

d2v_model_path = './enwiki_dbow/doc2vec.bin'
w2v_model_path = "./GoogleNews-vectors-negative300.bin.gz"

d2v_model = gensim.models.Doc2Vec.load(d2v_model_path)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
w2v_vocab = w2v_model.vocab.keys()


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop((args.img_size, args.img_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.dataset == "WP_MM":
        # trainset = MMFolder(image_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/images/train",
        #                     text_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/texts/train",
        #                     transform=transform_train, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        # testset = MMFolder(image_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/images/test",
        #                    text_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/texts/test",
        #                    transform=transform_test, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/wikipedia_dataset/images/all"))
        random.seed(args.dataset_seed)
        random.shuffle(classes)
        train_classes = sorted(classes[:int(len(classes) / 2)])
        test_classes = sorted(classes[int(len(classes) / 2):])
        #
        # train_classes = os.listdir("/deepo_data/GSP/CrossModal/wikipedia_dataset/images/train")
        # test_classes = os.listdir("/deepo_data/GSP/CrossModal/wikipedia_dataset/images/test")
        print('train_classes: ', train_classes)
        print('test_classes:', test_classes)

        trainset = MMDataset(classes=train_classes,
                             image_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/images/all",
                             text_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/texts/all",
                             transform=transform_train, dataset_name=args.dataset)
        testset = MMDataset(classes=test_classes,
                            image_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/images/all",
                            text_root="/deepo_data/GSP/CrossModal/wikipedia_dataset/texts/all",
                            transform=transform_test, dataset_name=args.dataset)

    elif args.dataset == "PS":
        classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/PascalSentence/image/all"))
        random.seed(args.dataset_seed)
        random.shuffle(classes)
        train_classes = sorted(classes[:int(len(classes) / 2)])
        test_classes = sorted(classes[int(len(classes) / 2):])

        print('train_classes: ', train_classes)
        print('test_classes:', test_classes)

        # trainset = MMFolder(image_root="/deepo_data/GSP/CrossModal/PascalSentence/image/ztrain",
        #                     text_root="/deepo_data/GSP/CrossModal/PascalSentence/sentence/ztrain",
        #                     transform=transform_train, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        # testset = MMFolder(image_root="/deepo_data/GSP/CrossModal/PascalSentence/image/ztest",
        #                    text_root="/deepo_data/GSP/CrossModal/PascalSentence/sentence/ztest",
        #                    transform=transform_test, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        trainset = MMDataset(classes=train_classes,
                             image_root="/deepo_data/GSP/CrossModal/PascalSentence/image/all",
                             text_root="/deepo_data/GSP/CrossModal/PascalSentence/sentence/all",
                             transform=transform_train, dataset_name=args.dataset)
        testset = MMDataset(classes=test_classes,
                            image_root="/deepo_data/GSP/CrossModal/PascalSentence/image/all",
                            text_root="/deepo_data/GSP/CrossModal/PascalSentence/sentence/all",
                            transform=transform_test, dataset_name=args.dataset)

    elif args.dataset == "XM":

        # trainset = MMFolder(image_root="/deepo_data/GSP/CrossModal/XMediaNet/image/train",
        #                     text_root="/deepo_data/GSP/CrossModal/XMediaNet/text/train",
        #                     transform=transform_train, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        # testset = MMFolder(image_root="/deepo_data/GSP/CrossModal/XMediaNet/image/test",
        #                    text_root="/deepo_data/GSP/CrossModal/XMediaNet/text/test",
        #                    transform=transform_test, dataset=args.dataset) if args.local_rank in [-1, 0] else None
        classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/XMediaNet/image/all"))
        random.seed(args.dataset_seed)
        random.shuffle(classes)
        train_classes = sorted(classes[:100])
        test_classes = sorted(classes[100:])
        trainset = MMDataset(classes=train_classes,
                             image_root="/deepo_data/GSP/CrossModal/XMediaNet/image/all",
                             text_root="/deepo_data/GSP/CrossModal/XMediaNet/text/all",
                             transform=transform_train, dataset_name=args.dataset)
        testset = MMDataset(classes=test_classes,
                            image_root="/deepo_data/GSP/CrossModal/XMediaNet/image/all",
                            text_root="/deepo_data/GSP/CrossModal/XMediaNet/text/all",
                            transform=transform_test, dataset_name=args.dataset)

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


class MMDataset(Dataset):
    def __init__(self, classes, image_root, text_root, transform=None, dataset_name='PS'):
        self.classes = sorted(classes)
        self.image_root = image_root
        self.text_root = text_root
        self.transform = transform
        self.dataset_name = dataset_name

        self.samples = self.get_all_samples(classes, self.image_root, self.text_root)

        cls_vectors = []
        for cls_word in classes:
            if cls_word in w2v_vocab:
                vec = w2v_model[cls_word]
            else:
                vec = d2v_model.infer_vector([cls_word])
            cls_vectors.append(vec)
        self.cls_embs = torch.FloatTensor(np.array(cls_vectors))
        # self.cls_embs = torch.rand([len(classes), 300])

    def get_all_samples(self, classes, image_root, text_root):
        instances = []
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for target_class in classes:
            target_dir_image = os.path.join(image_root, target_class)
            target_dir_text = os.path.join(text_root, target_class)
            class_index = class_to_idx[target_class]
            if not os.path.isdir(target_dir_image) or not os.path.isdir(target_dir_text):
                continue
            for root, _, image_names in sorted(os.walk(target_dir_image, followlinks=True)):
                for image_name in sorted(image_names):
                    text_name = image_name.split(".")[0] + ".txt"
                    # text_name = image_name.replace("jpg", "txt")
                    image_path = os.path.join(target_dir_image, image_name)
                    text_path = os.path.join(target_dir_text, text_name)
                    item = image_path, text_path, class_index
                    instances.append(item)
        return instances

    def sample_loader(self, image_path, text_path, dataset_name):
        img = Image.open(image_path).convert("RGB")
        if dataset_name in ['XM', 'WP_MM']:
            """ For dataset wikipedia &  XMediaNet: """
            with open(text_path, "r") as f:
                text = f.readlines()
            text = text[0]
        elif dataset_name in ['PS', 'CUB']:
            """ For dataset PascalSentence: """
            with open(text_path, "r") as f:
                texts = f.readlines()
            texts = [text.replace("\n", " ") for text in texts]
            text = "".join(texts)

        # if len(text.split(" ")) > 64:
        #     words = text.split(" ")[0:64]
        #     text = " ".join(words)

        return img, text  # a tuple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, text_path, target = self.samples[index]
        sample = self.sample_loader(image_path, text_path, self.dataset_name)
        img, text = sample
        if self.transform is not None:
            img = self.transform(img)

        # for vit image encoding
        img = np.array(img)

        # for w2v encoding
        # tokens = gensim.utils.simple_preprocess(text)
        # text_emb = torch.FloatTensor(np.array(d2v_model.infer_vector(tokens)))
        cls_emb = self.cls_embs[target]
        # text_emb = torch.rand([300])

        return img, text, cls_emb, target
