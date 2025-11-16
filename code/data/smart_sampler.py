import torch
import torch.nn.functional as F
from monai.transforms import MapTransform
import random

class SmartPosNegSampler(MapTransform):
    """
    MapTransform that produces `num_samples` crops per case.
    - If labels are all zeros -> returns num_samples random crops.
    - If lesions exist -> foreground_ratio of samples are taken near positive voxels (randomized),
      remaining are random crops (may or may not contain lesions).
    """
    def __init__(self, keys, roi_size, num_samples=4, foreground_ratio=0.75):
        super().__init__(keys)
        self.keys = keys
        if isinstance(roi_size[0], (list, tuple)):
            roi_size = roi_size[0]
        self.roi_d, self.roi_h, self.roi_w = tuple(int(x) for x in roi_size)
        self.num_samples = int(num_samples)
        self.foreground_ratio = float(foreground_ratio)

    def __call__(self, data):
        labels = (data["labels"] > 0).float()  # shape: (1, D, H, W)
        D, H, W = labels.shape[1:]

        def clamp_start(val, max_start):
            return max(0, min(val, max_start))

        def random_start():
            sd = random.randint(0, max(D - self.roi_d, 0))
            sh = random.randint(0, max(H - self.roi_h, 0))
            sw = random.randint(0, max(W - self.roi_w, 0))
            return sd, sh, sw

        # Get all positive voxel coordinates
        pos_coords = torch.nonzero(labels[0] > 0, as_tuple=False)  # (N, 3) -> z,y,x

        samples = []

        if pos_coords.numel() == 0:
            # No lesions: all random crops
            for _ in range(self.num_samples):
                sd, sh, sw = random_start()
                crop = {k: data[k][:, sd:sd + self.roi_d, sh:sh + self.roi_h, sw:sw + self.roi_w].clone()
                        for k in self.keys}
                crop['case_id'] = data.get('case_id', None)
                samples.append(crop)
            return samples

        # Lesions present
        num_foreground = int(self.num_samples * self.foreground_ratio)
        num_random = self.num_samples - num_foreground

        # Foreground-biased crops
        for _ in range(num_foreground):
            idx = random.randint(0, pos_coords.shape[0] - 1)
            cz, cy, cx = pos_coords[idx].tolist()
            # Random offset around voxel
            jz = random.randint(-self.roi_d // 4, self.roi_d // 4)
            jy = random.randint(-self.roi_h // 4, self.roi_h // 4)
            jx = random.randint(-self.roi_w // 4, self.roi_w // 4)
            czj, cyj, cxj = cz + jz, cy + jy, cx + jx
            sd = clamp_start(int(czj - self.roi_d // 2), D - self.roi_d)
            sh = clamp_start(int(cyj - self.roi_h // 2), H - self.roi_h)
            sw = clamp_start(int(cxj - self.roi_w // 2), W - self.roi_w)
            crop = {k: data[k][:, sd:sd + self.roi_d, sh:sh + self.roi_h, sw:sw + self.roi_w].clone()
                    for k in self.keys}
            crop['case_id'] = data.get('case_id', None)
            samples.append(crop)

        # Purely random crops
        for _ in range(num_random):
            sd, sh, sw = random_start()
            crop = {k: data[k][:, sd:sd + self.roi_d, sh:sh + self.roi_h, sw:sw + self.roi_w].clone()
                    for k in self.keys}
            crop['case_id'] = data.get('case_id', None)
            samples.append(crop)

        return samples

