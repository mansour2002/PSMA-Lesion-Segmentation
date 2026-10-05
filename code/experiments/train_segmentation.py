from typing import Union, Optional, Sequence


from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from models import UNETR8PS
from monai.networks.nets import SegResNet
from monai.data import decollate_batch
from monai.losses import DiceLoss

import numpy as np
import torch, random
import data
import optimizers

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


pl.seed_everything(42, workers=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SegmentationTrainer(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_dict: dict,
        frozen_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict

        if model_name.split("_")[0] == "unetr8ps":
            self.model = UNETR8PS(**model_dict)
        elif model_name == "segresnet":
            self.model = SegResNet(**model_dict)

        
        self.loss_function = DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=False)

        # Post-processing
        self.post_pred = AsDiscrete(sigmoid=True, threshold=0.5)
        self.post_label = AsDiscrete()

        # Metrics
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )

        # Tracking
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.metric_values = []
        self.epoch_loss_values = []
        self.val_dice_vals = []
        self.val_loss_vals = []
        self.val_num_items = []

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def stitch_patches(patches, z_starts, target_z_size):
        """Average overlapping axial patches [1, C, X, Y, roi_z] into one [1, C, X, Y, Z] column."""
        b, c, x, y, roi_z = patches[0].shape
        max_z = max(z + roi_z for z in z_starts)
        acc = torch.zeros((b, c, x, y, max_z), device=patches[0].device, dtype=patches[0].dtype)
        cnt = torch.zeros_like(acc)
        for patch, z in zip(patches, z_starts):
            acc[..., z:z + roi_z] += patch
            cnt[..., z:z + roi_z] += 1.0
        return (acc / (cnt + 1e-8))[..., :target_z_size]

    def stitched_forward(self, batch):
        """Run the network on all ASW patches of one volume and stitch predictions and labels
        along Z, so the loss is computed on the reconstructed column rather than per patch."""
        patches = batch["patches"]
        z_starts = [int(z.item()) for z in batch["z_starts"]]
        target_z = int(batch["original_shape"][-1].item())

        images = torch.stack([p["image"].squeeze(0) for p in patches]).to(self.device)
        labels = torch.stack([p["label"].squeeze(0) for p in patches]).to(self.device).float()
        outputs = self.forward(images)

        n = len(patches)
        pred = self.stitch_patches([outputs[i:i + 1] for i in range(n)], z_starts, target_z)
        label = self.stitch_patches([labels[i:i + 1] for i in range(n)], z_starts, target_z)
        return pred, label

    def training_step(self, batch, batch_idx):
        output, labels = self.stitched_forward(batch)
        batch_size = 1

        assert labels.min() >= 0 and labels.max() <= 1

        loss = self.loss_function(output, labels)

        # Logging
        self.log(
            "train/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.epoch_loss_values.append(loss.detach())

        return {"loss": loss}

    def on_train_epoch_end(self):
        if hasattr(self, 'epoch_loss_values') and len(self.epoch_loss_values) > 0:
            avg_loss = torch.stack(self.epoch_loss_values).mean()
            self.log(
                "train/dice_loss_avg",
                avg_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
            self.epoch_loss_values = []  
            
    def validation_step(self, batch, batch_idx):
        outputs, labels = self.stitched_forward(batch)
        batch_size = 1

        loss = self.loss_function(outputs, labels)

        # Compute dice score
        outputs_discrete = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels_discrete = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs_discrete, y=labels_discrete)
        dice = self.dice_metric.aggregate().item()

        # Logging
        self.log(
            "val/dice_loss_step",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        self.val_dice_vals.append(dice)
        self.val_loss_vals.append(loss.detach())
        self.val_num_items.append(len(outputs_discrete))

        return {
            "val_loss": loss,
            "val_number": len(outputs_discrete),
            "dice": dice,
        }

    def on_validation_epoch_end(self):
        if len(self.val_dice_vals) > 0:
            mean_val_dice = np.mean(self.val_dice_vals)
            mean_val_loss = torch.stack(self.val_loss_vals).mean()

            self.dice_metric.reset()

            if not torch.isnan(mean_val_loss):
                self.log(
                    "val/dice_loss_avg",
                    mean_val_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True
                )

            self.log(
                "val/dice_score_avg",
                mean_val_dice,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

            # Log hyperparameters
            self.logger.log_hyperparams(
                params={
                    "model": self.model_name,
                    **self.model_dict,
                    "loss": "DiceLoss",
                    "data": self.trainer.datamodule.json_path,
                    "batch_size": self.trainer.datamodule.batch_size,
                    "distribution": self.trainer.datamodule.dist,
                    "max_epochs": self.trainer.max_epochs,
                    "precision": self.trainer.precision,
                },
                metrics={"dice_loss": mean_val_loss, "dice_score": mean_val_dice},
            )

            if mean_val_dice > self.best_val_dice:
                self.best_val_dice = mean_val_dice
                self.best_val_epoch = self.current_epoch
                
            # Reset for next epoch
            self.val_dice_vals = []
            self.val_loss_vals = []
            self.val_num_items = []

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        batch_size = images.shape[0]
        outputs = self.forward(images)
        labels = labels.float()

        loss = self.loss_function(outputs, labels)

        # Compute dice score
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

    def test_epoch_end(self, outputs):
        mean_val_dice = torch.nanmean(self.dice_metric.get_buffer(), dim=0)
        print(f"Test Dice per class: {mean_val_dice}")
        print(f"Test Dice mean: {torch.mean(mean_val_dice)}")


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
