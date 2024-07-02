# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

from models.modeling import DualTransformers, XMModel

from utils.data_utils import get_loader, zero_shot_loader

from models.contrastive_loss import *

from utils.mAP import *

import time

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, CLIPModel

from accelerate import Accelerator
import logging
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__, log_level="INFO")


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


def save_model(args, model, accelerator):
    model.eval()
    output_dir = os.path.join(args.output_dir, args.dataset)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    accelerator.save_model(model, output_dir, safe_serialization=False)
    logger.info("Saved model checkpoint to [DIR: %s]", output_dir)


def load_model(args, model, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    output_dir = os.path.join(args.output_dir, args.dataset)
    path_to_checkpoint = os.path.join(output_dir, "pytorch_model.bin")
    unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def extract_features(args, model, data_loader, accelerator):
    model.eval()
    labels = []
    ins_img_embs, ins_text_embs = [], []
    cls_img_embs, cls_text_embs = [], []

    epoch_iterator = tqdm(data_loader,
                          desc="Extracting features... ",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=not accelerator.is_main_process)
    for step, batch in enumerate(epoch_iterator):
        img_encoding = batch[0].float()
        text_encoding = batch[1]
        y = batch[2]
        text_lens = batch[3]

        y = batch[2]

        with torch.no_grad():
            # text_lengths=None
            img_emb, text_emb, cls_img_emb, cls_text_emb, _, = model(img_encoding, text_encoding, text_lens)

            img_emb, text_emb, cls_img_emb, cls_text_emb, y = accelerator.gather_for_metrics(
                (img_emb, text_emb, cls_img_emb, cls_text_emb, y))

        labels.append(y.detach().cpu())
        ins_img_embs.append(img_emb.detach().cpu())
        ins_text_embs.append(text_emb.detach().cpu())
        cls_img_embs.append(cls_img_emb.detach().cpu())
        cls_text_embs.append(cls_text_emb.detach().cpu())

    labels = torch.cat(labels, dim=0).long()
    ins_img_embs = torch.cat(ins_img_embs, dim=0)
    ins_text_embs = torch.cat(ins_text_embs, dim=0)
    cls_img_embs = torch.cat(cls_img_embs, dim=0)
    cls_text_embs = torch.cat(cls_text_embs, dim=0)

    return labels, ins_img_embs, ins_text_embs, cls_img_embs, cls_text_embs


def extract_clip_features(args, model, data_loader):
    # logger.info("***** Extracting features *****")
    model.eval()
    labels = []
    ins_img_embs, ins_text_embs = [], []

    epoch_iterator = tqdm(data_loader,
                          desc="Extracting features... ",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        pixel_values = batch[0].to(args.device)
        input_ids = batch[1].to(args.device)

        y = batch[2].to(args.device)

        with torch.no_grad():
            # text_lengths=None
            img_emb = model.get_image_features(pixel_values)
            text_emb = model.get_text_features(input_ids=input_ids, attention_mask=(input_ids != 0))

        labels += y.detach().cpu().tolist()
        ins_img_embs.append(img_emb.detach().cpu())
        ins_text_embs.append(text_emb.detach().cpu())

    labels = torch.LongTensor(labels)
    ins_img_embs = torch.cat(ins_img_embs, dim=0)
    ins_text_embs = torch.cat(ins_text_embs, dim=0)

    cls_img_embs = None
    cls_text_embs = None

    return labels, ins_img_embs, ins_text_embs, cls_img_embs, cls_text_embs


def zs_validate(args, model, train_unseen_loader, test_unseen_loader, accelerator):
    # train_seen_loader, train_unseen_loader, test_seen_loader, test_unseen_loader = zero_shot_loader(args)
    tru_labels, tru_ins_img_embs, tru_ins_text_embs, \
        tru_cls_img_embs, tru_cls_text_embs = extract_features(args, model, train_unseen_loader, accelerator)
    teu_labels, teu_ins_img_embs, teu_ins_text_embs, \
        teu_cls_img_embs, teu_cls_text_embs = extract_features(args, model, test_unseen_loader, accelerator)

    if not accelerator.is_main_process:
        return None

    it_mAP_ins = compute_mAP(args, tru_ins_text_embs, teu_ins_img_embs, tru_labels, teu_labels)
    ti_mAP_ins = compute_mAP(args, tru_ins_img_embs, teu_ins_text_embs, tru_labels, teu_labels)
    it_mAP_cls = compute_mAP_dis(args, tru_cls_text_embs, teu_cls_img_embs, tru_labels, teu_labels)
    ti_mAP_cls = compute_mAP_dis(args, tru_cls_img_embs, teu_cls_text_embs, tru_labels, teu_labels)

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


    results = np.array([[it_mAP_ins, it_mAP_cls, it_mAP],
                        [ti_mAP_ins, ti_mAP_cls, ti_mAP],
                        [(it_mAP_ins + ti_mAP_ins) / 2, (it_mAP_cls + ti_mAP_cls) / 2,  (it_mAP + ti_mAP) / 2]])

    return results


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

        text_encoding = batch[1].to(args.device)


        img_encoding = batch[0]
        img_encoding = [img for img in img_encoding]
        img_encoding = feature_extractor(images=img_encoding, return_tensors="pt")["pixel_values"]
        img_encoding = img_encoding.to(args.device)

        y = batch[2].to(args.device)

        with torch.no_grad():

            img_emb, text_emb, cls_img_emb, cls_text_emb, _, = model(img_encoding, text_encoding)

            preds = 0
        if len(all_preds) == 0:

            all_preds.append(preds)
            all_label.append(y.detach().cpu().numpy())
            all_img_emb.append(img_emb.detach().cpu().numpy())
            all_text_emb.append(text_emb.detach().cpu().numpy())
            cls_img_embs.append(cls_img_emb.detach().cpu().numpy())
            cls_text_embs.append(cls_text_emb.detach().cpu().numpy())

        else:
            all_preds[0] = np.append(
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

    it_mAP_ins = compute_mAP(args, all_text_emb, all_img_emb, all_label, all_label)
    ti_mAP_ins = compute_mAP(args, all_img_emb, all_text_emb, all_label, all_label)
    it_mAP_cls = compute_mAP(args, cls_text_embs, cls_img_embs, all_label, all_label)
    ti_mAP_cls = compute_mAP(args, cls_img_embs, cls_text_embs, all_label, all_label)
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


def train(args, model, train_seen_loader, optimizer, scheduler, accelerator):
    """ Train the model """

    loss_inter = FGCL(margin1=args.margin, margin2=args.margin2, wp_weight=args.wp_weight)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_mAP = 0, 0

    model.train()
    for epoch in range(args.num_epochs):
        epoch_iterator = tqdm(train_seen_loader,
                              desc="Training (X / X Epochs) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=not accelerator.is_main_process)
        for step, batch in enumerate(epoch_iterator):

            img_encoding = batch[0].float()
            text_encoding = batch[1]

            y = batch[2]
            text_lens = batch[3]


            img_emb, text_emb, zs_img_emb, zs_text_emb, fgla_loss = model(img_encoding, text_encoding, text_lens)

            fgcl_loss = 0.5 * loss_inter(y, img_emb, text_emb) + 0.5 * loss_inter(y, text_emb, img_emb)

            loss = fgcl_loss + fgla_loss.mean()

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.update(loss.item())

            epoch_iterator.set_description(
                "Training (%d / %d Epochs)(loss=%2.5f)" % (epoch + 1, args.num_epochs, losses.val))

    if accelerator.is_main_process:
        save_model(args, model, accelerator)

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
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight delay if we apply some.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset_seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--eval_only', action='store_true',
                        help='Whether to train or validate the model.')

    parser.add_argument('--margin', type=float, default=0.7)
    parser.add_argument('--margin2', type=float, default=0.3)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--temp', type=float, default=2.0)
    parser.add_argument('--kd_lr', type=float, default=1e-4)

    parser.add_argument('--num_v', type=float, default=0.1, help='ratio of image patches chosen for FGCL.')
    parser.add_argument('--num_t', type=float, default=0.1, help='ratio of text tokens chosen for FGCL.')

    parser.add_argument('--wp_weight', type=float, default=0.1, help='coefficient for weak positive loss')

    args = parser.parse_args()

    accelerator = Accelerator()

    # Set seed
    set_seed(args.seed)

    # Model & Tokenizer Setup
    train_seen_loader, train_unseen_loader, \
        test_seen_loader, test_unseen_loader = zero_shot_loader(args)

    model = DualTransformers(margin=args.margin, margin2=args.margin2,
                             dataset=args.dataset, temp=args.temp,
                             num_v=args.num_v, num_t=args.num_t)

    model.to(accelerator.device)

    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_epochs * len(train_seen_loader))

    model, optimizer, train_seen_loader, scheduler, train_unseen_loader, test_unseen_loader = accelerator.prepare(
        model, optimizer, train_seen_loader, scheduler, train_unseen_loader, test_unseen_loader
    )

    # Training
    if not args.eval_only:
        time_start = time.time()
        train(args, model, train_seen_loader, optimizer, scheduler, accelerator)
        time_end = time.time()
        logger.info('Training time cost: %2.1f minutes.' % ((time_end - time_start) / 60))

    # Validating
    accelerator.wait_for_everyone()
    # model = load_model(args, model, accelerator)
    results = zs_validate(args, model, train_unseen_loader, test_unseen_loader, accelerator)

    if accelerator.is_main_process:
        output_dir = os.path.join(args.output_dir, args.dataset, 'results')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, str(args.dataset_seed) + '.json'), results)


if __name__ == "__main__":
    main()
