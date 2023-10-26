# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

       # self.models["encoder"] = networks.ResnetEncoder(
        #    self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"] = networks.mpvit_small()
        self.models["encoder"].num_ch_enc = [64,128,216,288,288]#[64, 128, 192, 256, 256]#
        self.models["encoder"].to(self.device)
        #self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc)
            #self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.use_posegt:
            self.models["trans_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["trans_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["trans_encoder"].parameters())

            self.models["trans"] = networks.TransDecoder(
                self.models["trans_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

            self.models["trans"].to(self.device)
            self.parameters_to_train += list(self.models["trans"].parameters())


        #self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.params = [ {
            "params":self.parameters_to_train, 
            "lr": 1e-4#self.opt.learning_rate
            #"weight_decay": 0.01
            },
            {
            "params": list(self.models["encoder"].parameters()), 
           "lr": self.opt.learning_rate
            #"weight_decay": 0.01
            } ]
            #self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_optimizer = optim.AdamW(self.params)
        #self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
		self.model_optimizer,0.9)
        #self.model_lr_scheduler = optim.lr_scheduler.StepLR(
         #   self.model_optimizer, self.opt.scheduler_step_size, 0.5)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyu": datasets.NYURAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            if self.epoch <10:
                iiter=1
            else:
                iiter=self.opt.iiters
            self.bestpp={}
            for i in range(iiter):
                outputs, losses = self.process_batch(inputs)
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)
        if self.opt.use_posegt:
            self.generate_images_posegt(inputs,outputs)

        if self.use_pose_net:
            self.predict_poses(inputs, outputs)

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def generate_images_posegt(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        scale = 0
        disp = outputs[("disp", 0)]
        source_scale = 0
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        for frame_id in self.opt.frame_ids[1:]:

            raw_img = inputs["color_aug", frame_id, 0]
            tgt_img = inputs["color_aug", 0, 0]
            if frame_id < 0:
                pose_inputs = [raw_img, tgt_img]
            else:
                pose_inputs = [tgt_img, raw_img]

            pose_inputs = [self.models["trans_encoder"](torch.cat(pose_inputs, 1))]
            translation, scalor = self.models["trans"](pose_inputs)

            pose_gt = inputs["pose_gt",frame_id].cuda().clone()# get generated coarse COLMAP pose
            #rotation from coarse COLMAP pose, translation from both COLMAP and networks
            T = transformation_from_gtmsrtpose(
                    pose_gt, scalor[:,0],translation[:, 0], invert=(frame_id < 0))

            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)

            source_img = inputs[("color", frame_id, 0)]
            outputs[("irw_img", frame_id, 0)] = F.grid_sample(
                source_img, pix_coords,
                padding_mode="border", align_corners=True)

    def predict_poses(self, inputs, outputs):
        """Predict poses between input frames for monocular sequences.
        """
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if self.opt.use_posegt:
                        pose_img = outputs["irw_img", f_i, 0].clone()
                    else:
                        pose_img = pose_feats[f_i]
                    if f_i < 0:
                        pose_inputs = [pose_img, pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_img]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    axisangle, translation, scalor = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                source_img = inputs[("color", frame_id, source_scale)]
                if self.opt.use_posegt:
                    source_img = outputs[("irw_img", frame_id, 0)]
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_img,
                    pix_coords,
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = source_img

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)/(target+0.01)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        rawcolor = inputs[("color", 0, 0)]
        usepp = self.opt.selfpp
        usegtp = self.opt.use_posegt
        

        if not self.opt.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, 0)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred,rawcolor))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        if usepp:
            if self.bestpp=={}:
                disp_best=outputs[("disp", 0)]
                reproj_loss_min=torch.ones(disp_best.shape).cuda()*10
            else:
                disp_best=self.bestpp["disp"]
                reproj_loss_min=self.bestpp["error"]
                
        for scale in self.opt.scales:
            loss = 0
            loss_pose = 0
            reprojection_losses = []
            sl_loss =[]
            sd_loss = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                ################### pp #############
                if usegtp:
                    pred_select = outputs[("irw_img", frame_id, 0)]
                    sl_loss.append(self.compute_reprojection_loss(pred_select, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)


            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)


            ###########
            if usegtp:
                sl_loss = torch.cat(sl_loss, 1)
                sl_losses = torch.cat((identity_reprojection_loss, sl_loss), dim=1)
                to_optimise2, _ = torch.min(sl_losses, dim=1, keepdim=True)
                loss_pose += to_optimise2.mean()*self.opt.www

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            ################### pp #############
            if usepp:
                #selfpp_errors = reprojection_losses#torch.cat(selfpp_errors,1)
                error_map= to_optimise
                disp_best = torch.where(error_map < reproj_loss_min, disp, disp_best)
                reproj_loss_min, _ = torch.cat([error_map, reproj_loss_min], dim=1).min(dim=1, keepdim=True)
                
            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, rawcolor)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            if usegtp:
                loss += loss_pose
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        ############################# compute pp loss ##############################
        if usepp:
            if self.epoch < 20:
                www = self.opt.wpp * self.epoch /20
            else:
                www = self.opt.wpp
            selfpp_loss = self.comp_selfpp_losses(disp_best,outputs)#+self.comp_selfpp_losses(disp_best,outputs)
            total_loss += selfpp_loss*www
            self.bestpp["disp"]=disp_best
            self.bestpp["error"]=reproj_loss_min

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses


    def comp_selfpp_losses(self, disp, outputs):
        pp_loss = 0
        #so just test in the future
        #print(disp_best.shape)
        disp_best = disp.detach()
        mask = outputs["identity_selection/{}".format(0)].detach()
        for scale in self.opt.scales:
            disp_pred = outputs[("disp", scale)]
            disp_pred = F.interpolate(
                    disp_pred, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            pp_loss += torch.log(torch.abs(disp_best - disp_pred) + 1).mean()#/ (2 ** scale))*mask
        return pp_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
