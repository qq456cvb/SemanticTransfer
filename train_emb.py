from models.pointnet import PointNetDenseCls
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import os
from datasets import kpnet
import logging
from itertools import combinations
import numpy as np
from tqdm import tqdm


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
        
        
class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError
    

# reference: https://github.com/adambielski/siamese-triplet
class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(
            labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(
            labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:,
                                                            0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[
            :len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs
    

# reference: https://github.com/adambielski/siamese-triplet
class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector, mean_distance=None):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        if mean_distance is not None:
            self.mean_distance = mean_distance[0].cuda()
        else:
            self.mean_distance = None

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] -
                         embeddings[positive_pairs[:, 1]]).pow(2).sum(1)

        labels_1 = tuple(target[negative_pairs[:, 0]].tolist())
        labels_2 = tuple(target[negative_pairs[:, 1]].tolist())
        label_pair = (labels_1, labels_2)

        if self.mean_distance is not None:
            negative_loss = F.relu(
                self.mean_distance[label_pair] - ((embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                    1) + 1e-6).sqrt()).pow(2)
        else:
            negative_loss = F.relu(
                self.margin - ((embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                    1) + 1e-6).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    
    train_dataset = kpnet.KeypointDataset(cfg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    
    model = PointNetDenseCls(feature_transform=True, cfg=cfg).cuda()
    
    logger.info('Start training on 3D embeddings')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )
    criterion = OnlineContrastiveLoss(1., HardNegativePairSelector())
    
    meter = AverageMeter()
    for epoch in range(cfg.max_epoch + 1):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for i, (pc, kp_idxs) in enumerate(train_iter):
            pc, kp_idxs = pc.cuda(), kp_idxs.cuda()
            outputs = model(pc.transpose(1, 2))
            
            embeddings = []
            labels = []
            for i in range(cfg.batch_size):
                embedding_model = outputs[i]
                keypoints = kp_idxs[i]
                for idx in range(len(keypoints)):
                    kp_idx = keypoints[idx]
                    if kp_idx < 0:
                        continue
                    
                    embedding_kp = embedding_model[kp_idx]
                    embeddings.append(embedding_kp)
                    labels.append(idx)
                    
            embeddings = torch.stack(embeddings)
            labels = torch.tensor(labels).cuda()
            
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
            
        logger.info(
                f'Epoch: {epoch}, Average Train loss: {meter.avg}'
            )
        
        torch.save(model.state_dict(), f'epoch{epoch}.pth')


if __name__ == '__main__':
    main()
    