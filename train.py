# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm.notebook import tqdm
# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import DualTransformers

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, zero_shot_loader

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertTokenizer, ViTFeatureExtractor, BertModel
from models.contrastive_loss import *

import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
from utils.dist_util import get_world_size
from utils.mAP import *

import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# bert = BertModel.from_pretrained('bert-base-uncased')

logger = logging.getLogger("__name__")
logging.getLogger().setLevel(logging.INFO)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model.eval()
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def load_model(args, model):
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    state_dict = torch.load(model_checkpoint)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


def setup(args):
    # if args.dataset == "PS":
    #     classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/PascalSentence/image/all"))
    #     random.seed(args.dataset_seed)
    #     random.shuffle(classes)
    #     train_classes = sorted(classes[:int(len(classes) / 2)])
    # elif args.dataset == "WP_MM":
    #     classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/wikipedia_dataset/images/all"))
    #     random.seed(args.dataset_seed)
    #     random.shuffle(classes)
    #     train_classes = sorted(classes[:int(len(classes) / 2)])
    # elif args.dataset == "XM":
    #     classes = sorted(os.listdir("/deepo_data/GSP/CrossModal/XMediaNet/image/all"))
    #     random.seed(args.dataset_seed)
    #     random.shuffle(classes)
    #     train_classes = sorted(classes[:100])

    model = DualTransformers(margin=args.margin, margin2=args.margin2, dataset=args.dataset, temp=args.temp)
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def extract_features(args, model, data_loader):
    # logger.info("***** Extracting features *****")
    model.eval()
    labels = []
    ins_img_embs, ins_text_embs = [], []
    cls_img_embs, cls_text_embs = [], []

    epoch_iterator = tqdm(data_loader,
                          desc="Extracting features... ",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        img_encoding = batch[0].float().to(args.device)
        text_encoding = batch[1].float().to(args.device)
        # texts = batch[4]
        # print(str(texts[0]) + ',')

        # img_encoding = batch[0]
        # img_encoding = [img for img in img_encoding]
        # img_encoding = feature_extractor(images=img_encoding, return_tensors="pt")["pixel_values"]
        # img_encoding = img_encoding.to(args.device)

        # text_tokens = tokenizer(batch[1], return_tensors='pt', padding="max_length", max_length=96, truncation=True)
        # input_ids = text_tokens['input_ids'].to(args.device)
        # attention_mask = text_tokens['attention_mask'].to(args.device)
        # text_encoding = (input_ids, attention_mask)

        y = batch[2].to(args.device)

        with torch.no_grad():
            # text_lengths=None
            img_emb, text_emb, cls_img_emb, cls_text_emb, _, = model(img_encoding, text_encoding)

        labels += y.detach().cpu().tolist()
        ins_img_embs += img_emb.detach().cpu().tolist()
        ins_text_embs += text_emb.detach().cpu().tolist()
        cls_img_embs += cls_img_emb.detach().cpu().tolist()
        cls_text_embs += cls_text_emb.detach().cpu().tolist()

    labels = torch.LongTensor(labels)
    ins_img_embs = torch.FloatTensor(ins_img_embs)
    ins_text_embs = torch.FloatTensor(ins_text_embs)
    cls_img_embs = torch.FloatTensor(cls_img_embs)
    cls_text_embs = torch.FloatTensor(cls_text_embs)

    return labels, ins_img_embs, ins_text_embs, cls_img_embs, cls_text_embs


def zs_validate(args, model, train_unseen_loader, test_unseen_loader):
    # train_seen_loader, train_unseen_loader, test_seen_loader, test_unseen_loader = zero_shot_loader(args)
    tru_labels, tru_ins_img_embs, tru_ins_text_embs, tru_cls_img_embs, tru_cls_text_embs = extract_features(args, model,
                                                                                                            train_unseen_loader)
    teu_labels, teu_ins_img_embs, teu_ins_text_embs, teu_cls_img_embs, teu_cls_text_embs = extract_features(args, model,
                                                                                                            test_unseen_loader)

    # save features for t-SNE.
    ins_img_embs = torch.cat([tru_ins_img_embs.reshape(10, -1, tru_ins_img_embs.shape[-1]), teu_ins_img_embs.reshape(10, -1, tru_ins_img_embs.shape[-1])], dim=1).reshape(-1, tru_ins_img_embs.shape[-1])
    ins_text_embs = torch.cat([tru_ins_text_embs.reshape(10, -1, tru_ins_text_embs.shape[-1]), teu_ins_text_embs.reshape(10, -1, tru_ins_text_embs.shape[-1])], dim=1).reshape(-1, tru_ins_text_embs.shape[-1])
    cls_img_embs = torch.cat([tru_cls_img_embs.reshape(10, -1, tru_cls_img_embs.shape[-1]), teu_cls_img_embs.reshape(10, -1, tru_cls_img_embs.shape[-1])], dim=1).reshape(-1, tru_cls_img_embs.shape[-1])
    cls_text_embs = torch.cat([tru_cls_text_embs.reshape(10, -1, tru_cls_text_embs.shape[-1]), teu_cls_text_embs.reshape(10, -1, tru_cls_text_embs.shape[-1])], dim=1).reshape(-1, tru_cls_text_embs.shape[-1])

    np.save('./visualizations/trained/ins_img_embs.npy', ins_img_embs)
    np.save('./visualizations/trained/ins_text_embs.npy', ins_text_embs)
    np.save('./visualizations/trained/cls_img_embs.npy', cls_img_embs)
    np.save('./visualizations/trained/cls_text_embs.npy', cls_text_embs)

    it_mAP_ins = compute_mAP(args, tru_ins_text_embs, teu_ins_img_embs, tru_labels, teu_labels)
    ti_mAP_ins = compute_mAP(args, tru_ins_img_embs, teu_ins_text_embs, tru_labels, teu_labels)
    it_mAP_cls = compute_mAP_dis(args, tru_cls_text_embs, teu_cls_img_embs, tru_labels, teu_labels)
    ti_mAP_cls = compute_mAP_dis(args, tru_cls_img_embs, teu_cls_text_embs, tru_labels, teu_labels)
    #
    it_mAP = compute_mAP_2(args, tru_ins_text_embs, tru_cls_text_embs, teu_ins_img_embs, teu_cls_img_embs,
                           tru_labels, teu_labels)
    ti_mAP = compute_mAP_2(args, tru_ins_img_embs, tru_cls_img_embs, teu_ins_text_embs, teu_cls_text_embs,
                           tru_labels, teu_labels)

    logger.info("\n")
    logger.info("Testing Results")
    logger.info("it_mAP: %2.3f, %2.3f, %2.3f" % (it_mAP_ins, it_mAP_cls, it_mAP))
    logger.info("ti_mAP: %2.3f, %2.3f, %2.3f" % (ti_mAP_ins, ti_mAP_cls, ti_mAP))
    logger.info("avg_mAP: %2.3f, %2.3f, %2.3f" % (
        ((it_mAP_ins + ti_mAP_ins) / 2), ((it_mAP_cls + ti_mAP_cls) / 2),
        (it_mAP + ti_mAP) / 2))
    #
    try:
        results = np.load(args.dataset+'map.npy').tolist()
    except:
        results = []

    r = [it_mAP_ins, ti_mAP_ins, it_mAP_cls, ti_mAP_cls, it_mAP, ti_mAP]
    results.append(r)
    np.save(args.dataset+'map.npy', results)

    return it_mAP


def valid(args, model):
    train_loader, test_loader = get_loader(args)

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    all_preds, all_label = [], []
    all_img_emb, all_text_emb = [], []
    cls_img_embs, cls_text_embs = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... ",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(epoch_iterator):

        # img_encoding = batch[0].to(args.device)
        text_encoding = batch[1].to(args.device)

        # texts = batch[4]
        # print(str(texts[0]) + ',')

        img_encoding = batch[0]
        img_encoding = [img for img in img_encoding]
        img_encoding = feature_extractor(images=img_encoding, return_tensors="pt")["pixel_values"]
        img_encoding = img_encoding.to(args.device)

        # text_tokens = tokenizer(batch[1], return_tensors='pt', padding="max_length", max_length=96, truncation=True)
        # input_ids = text_tokens['input_ids'].to(args.device)
        # attention_mask = text_tokens['attention_mask'].to(args.device)
        # text_encoding = (input_ids, attention_mask)

        y = batch[2].to(args.device)

        with torch.no_grad():
            # text_lengths=None
            img_emb, text_emb, cls_img_emb, cls_text_emb, _, = model(img_encoding, text_encoding)

            # eval_loss = loss_fct(logits, y)
            # eval_losses.update(eval_loss.item())
            # preds = torch.argmax(logits, dim=-1)
            preds = 0
        if len(all_preds) == 0:
            # all_preds.append(preds.detach().cpu().numpy())
            all_preds.append(preds)
            all_label.append(y.detach().cpu().numpy())
            all_img_emb.append(img_emb.detach().cpu().numpy())
            all_text_emb.append(text_emb.detach().cpu().numpy())
            cls_img_embs.append(cls_img_emb.detach().cpu().numpy())
            cls_text_embs.append(cls_text_emb.detach().cpu().numpy())

        else:
            all_preds[0] = np.append(
                # all_preds[0], preds.detach().cpu().numpy(), axis=0
                all_preds[0], preds
            )

            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_img_emb[0] = np.append(
                all_img_emb[0], img_emb.detach().cpu().numpy(), axis=0
            )
            all_text_emb[0] = np.append(
                all_text_emb[0], text_emb.detach().cpu().numpy(), axis=0
            )
            cls_img_embs[0] = np.append(
                cls_img_embs[0], cls_img_emb.detach().cpu().numpy(), axis=0
            )
            cls_text_embs[0] = np.append(
                cls_text_embs[0], cls_text_emb.detach().cpu().numpy(), axis=0
            )

        epoch_iterator.set_description("Validating... ")

    all_preds, all_label = all_preds[0], all_label[0]
    all_img_emb, all_text_emb = all_img_emb[0], all_text_emb[0]
    cls_img_embs, cls_text_embs = cls_img_embs[0], cls_text_embs[0]

    cls_img_embs = torch.from_numpy(cls_img_embs)
    cls_text_embs = torch.from_numpy(cls_text_embs)
    all_label = torch.from_numpy(all_label)
    all_img_emb = torch.from_numpy(all_img_emb)
    all_text_emb = torch.from_numpy(all_text_emb)

    # print("img emb: ", all_img_emb[0], all_img_emb[-1])
    # print("text emb: ", all_text_emb[0], all_img_emb[-1])
    # accuracy = simple_accuracy(all_preds, all_label)

    it_mAP_ins = compute_mAP(args, all_text_emb, all_img_emb, all_label, all_label)
    ti_mAP_ins = compute_mAP(args, all_img_emb, all_text_emb, all_label, all_label)
    it_mAP_cls = compute_mAP_dis(args, cls_text_embs, cls_img_embs, all_label, all_label)
    ti_mAP_cls = compute_mAP_dis(args, cls_img_embs, cls_text_embs, all_label, all_label)
    #
    it_mAP = compute_mAP_2(args, all_text_emb, cls_text_embs, all_img_emb, cls_img_embs,
                           all_label, all_label)
    ti_mAP = compute_mAP_2(args, all_img_emb, cls_img_embs, all_text_emb, cls_text_embs,
                           all_label, all_label)

    logger.info("\n")
    logger.info("Testing Results")
    logger.info("it_mAP: %2.3f, %2.3f, %2.3f" % (it_mAP_ins, it_mAP_cls, it_mAP))
    logger.info("ti_mAP: %2.3f, %2.3f, %2.3f" % (ti_mAP_ins, ti_mAP_cls, ti_mAP))
    logger.info("avg_mAP: %2.3f, %2.3f, %2.3f" % (
        ((it_mAP_ins + ti_mAP_ins) / 2), ((it_mAP_cls + ti_mAP_cls) / 2),
        (it_mAP + ti_mAP) / 2))
    #
    return it_mAP


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # Prepare dataset
    # train_loader, test_loader = get_loader(args)
    train_seen_loader, train_unseen_loader, test_seen_loader, test_unseen_loader = zero_shot_loader(args)

    # Prepare optimizer and scheduler
    # vit_params = list(map(id, model.IE.vit.parameters()))
    # bert_params = list(map(id, model.TE.bert.parameters()))
    ie_params = list(map(id, model.IE.vgg.parameters()))
    te_params = list(map(id, model.TE.base_fc.parameters()))
    # kd_params = list(map(id, model.img_kd.parameters()))
    ae_params = list(map(id, model.ae.parameters()))
    # resnet_params = list(map(id, model.resnet.parameters()))

    # IE_params = list(map(id, model.IE.parameters()))
    # TE_params = list(map(id, model.TE.parameters()))

    head_params = filter(lambda p: id(p) not in ie_params + te_params + ae_params, model.parameters())
    # head_params = filter(lambda p: id(p) not in ie_params + ae_params, model.parameters())

    optimizer = torch.optim.AdamW([
        # {'params': model.IE.vit.parameters(), 'lr': args.learning_rate},
        # {'params': model.TE.bert.parameters(), 'lr': args.learning_rate},
        {'params': model.IE.vgg.parameters(), 'lr': args.learning_rate},
        {'params': model.TE.base_fc.parameters(), 'lr': args.learning_rate},
        # {'params': model.IE.parameters(), 'lr': args.learning_rate},
        # {'params': model.TE.parameters(), 'lr': args.learning_rate},
        {'params': model.ae.parameters(), 'lr': args.learning_rate * 10},
        {'params': head_params, 'lr': args.learning_rate * 10},
    ])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.warmup_steps)
    # loss_func = ContrastiveLoss(margin=args.margin, margin2=args.margin2, temp=args.temp)

    # loss_zs = KLLoss()
    # loss_zs = JSLoss()
    # loss_zs = MMDLoss()
    # loss_zs = SimLoss()
    # loss_inter = nn.CrossEntropyLoss()
    # loss_inter = InterContrastiveLoss(margin=args.margin)
    # loss_inter = TripletLoss(margin=args.margin)
    loss_inter = SelectiveTripletsLoss(margin1=args.margin, margin2=args.margin2)
    # loss_inter = AllTripletsLoss(margin1=args.margin, margin2=args.margin2)
    # loss_inter.to(args.device)
    # optimizer2 = torch.optim.SGD(loss_inter.parameters(), lr=0.1)
    # if args.fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    # if args.local_rank != -1:
    #     model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # else:
    #     model = nn.DataParallel(model)
    model = nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_mAP = 0, 0

    # model.train()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_iterator = tqdm(train_seen_loader,
                              desc="Training (X / X Epochs) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # batch: (img, text), label,
            # batch: (img, img_copy, text), label

            img_encoding = batch[0].float().to(args.device)
            text_encoding = batch[1].float().to(args.device)

            # img_encoding = batch[0]
            # img_encoding = [img for img in img_encoding]
            # img_encoding = feature_extractor(images=img_encoding, return_tensors="pt")['pixel_values']
            # img_encoding = img_encoding.to(args.device)

            # text_tokens = tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True)
            # input_ids = text_tokens['input_ids'].to(args.device)
            # attention_mask = text_tokens['attention_mask'].to(args.device)
            # text_encoding = (input_ids, attention_mask)

            y = batch[2].to(args.device)
            # con_loss, zs_loss = model(input_ids, labels=y)
            # img_emb, text_emb, zs_img_emb, zs_text_emb, ae_loss = model(img_encoding, text_encoding, text_lengths, labels=y)
            # text_lengths = None
            img_emb, text_emb, zs_img_emb, zs_text_emb, ae_loss = model(img_encoding, text_encoding)

            con_loss1 = loss_inter(y, img_emb, text_emb)
            # con_loss1 = 0.5 * loss_inter(img_emb, y) + 0.5 * loss_inter(text_emb, y)
            # con_loss1 = loss_inter(y, text_emb, img_emb)
            # con_loss1 = 0.5 * loss_inter(y, img_emb, text_emb) + 0.5 * loss_inter(y, text_emb, img_emb)
            # con_loss2 = loss_zs(cls_preds, zs_img_emb, zs_text_emb)

            # text_embs as anchors
            # con_loss1 = loss_inter(y, text_emb, img_emb)

            loss = con_loss1 + 0.1 * ae_loss.mean()
            # loss = con_loss1
            # loss = 0.1 * ae_loss.mean()
            # print("vae_loss: ", vae_loss)
            # print("con_loss2: ", con_loss2)
            # loss = 0.01 * con_loss2
            # loss = con_loss1

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Epochs)(loss=%2.5f)" % (epoch + 1, args.num_epochs, losses.val))

        scheduler.step()
        # valid(args, model)
        # zs_validate(args, model, train_unseen_loader, test_unseen_loader)

    save_model(args, model)

    if args.local_rank in [-1, 0]:
        writer.close()
    # logger.info("mAP: \t%f" % mAP)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["WP", "PS", "XM"],
                        default="PS",
                        help="Which downstream task.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight delay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset_seed', type=int, default=5,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--eval_only', action='store_true',
                        help='Whether to train or validate the model.')

    parser.add_argument('--margin', type=float, default=0.7)
    parser.add_argument('--margin2', type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--temp', type=float, default=2.0)
    parser.add_argument('--kd_lr', type=float, default=1e-4)

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(seconds=30))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup

    args, model = setup(args)

    # Training
    if not args.eval_only:
        time_start = time.time()
        train(args, model)
        time_end = time.time()
        logger.info('Training time cost: %2.1f minutes.' % ((time_end - time_start) / 60))

    # Validating
    model = load_model(args, model)
    train_seen_loader, train_unseen_loader, test_seen_loader, test_unseen_loader = zero_shot_loader(args)
    # valid(args, model)
    zs_validate(args, model, train_unseen_loader, test_unseen_loader)


if __name__ == "__main__":
    main()
