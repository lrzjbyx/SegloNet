import torch


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, inputs,instances,target):
        assert len(inputs.shape) == 4, 'logits should be 4-ndim'
        device = inputs.device
        bs = inputs.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = instances["heatmap"]
        # [num_kps] -> [B, num_kps]
        kps_weights = instances["kps_weights"]

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(inputs, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs

        return loss
