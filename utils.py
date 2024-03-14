import math
import os
from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torchvision import transforms as T, utils
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from ema_pytorch import EMA
from accelerate import Accelerator
from diffusers import AutoencoderKL

from misc import num_to_groups
from misc import convert_image_to_fn
from misc import exists
from misc import has_int_squareroot
from misc import cycle
from misc import divisible_by


class FIDEvaluation(object):
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )

            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )

            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            if os.path.exists(path):
                np.savez_compressed(path, m2=m2, s2=s2)
                self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            fake_samples = self.sampler.sample(batch_size=batch, clip_denoising=True)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()

        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)
        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


class ImageData(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        augment_horizontal_flip=False,
        convert_image_to=None,
        in_mem=False,
        verbose=True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.in_mem = in_mem
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        # file_nms = [path for path in os.listdir(folder) if path.split('.')[1] in exts]
        # file_nms = sorted(file_nms, key=lambda x: int(x.split('.')[0]))
        # self.paths = [os.path.join(folder, fn) for fn in file_nms]

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if exists(convert_image_to)
            else nn.Identity()
        )
        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),  # [H, W, C] -> [H1, W1, C]
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),  # [H, W, C] -> [L, L, C]
                T.ToTensor(),  # [0, 255] -> [0, 1]; [L, L, C] -> [C, L, L]
            ]
        )

        if in_mem:
            idx_iter = (
                tqdm(range(0, len(self.paths)), desc="data loading")
                if verbose
                else range(0, len(self.paths))
            )
            self.data = [self._read_img(j) for j in idx_iter]

    def __len__(self):
        return len(self.paths)

    def _read_img(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

    def __getitem__(self, index):
        if self.in_mem:
            return self.data[index]
        else:
            return self._read_img(index)


class VecData(Dataset):
    def __init__(self, folder):
        super().__init__()

        self._data_arr = np.loadtxt(os.path.join(folder, "train.txt"))

    def __len__(self):
        return self._data_arr.shape[0]

    def __getitem__(self, index):
        tensor = torch.FloatTensor(self._data_arr[index])
        return tensor.reshape(-1, 1, 1)


class LatentVarData(Dataset):
    def __init__(self, folder, scaling_factor, in_mem=True, verbose=True):
        super().__init__()

        self._scaling_factor = scaling_factor
        self._in_mem = in_mem
        self._paths = [path for path in Path(f"{folder}").glob("**/*.pt")]
        # file_names = [path for path in os.listdir(folder) if path.endswith('.pt')]
        # file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
        # self._paths = [os.path.join(folder, fn) for fn in file_names]

        if in_mem:
            total = len(self._paths)
            progress = (
                tqdm(range(0, total), desc="data loading")
                if verbose
                else range(0, total)
            )
            self._data = [torch.load(self._paths[idx]) for idx in progress]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        if self._in_mem:
            tensor = self._data[index]
        else:
            tensor = torch.load(self._paths[index])

        if len(tensor.shape) == 4:
            mean, var = tensor[0], tensor[1]
            sample = mean + var * torch.randn_like(var)
            return sample * self._scaling_factor
        else:
            return tensor


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        test_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./save",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.0,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        data_type="image",
        io_workers=None,
        in_memory=False,
        pretrained_vae_dir=None,
        scaling_factor,
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters
        assert has_int_squareroot(
            num_samples
        ), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.test_bsz = test_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (
            (train_batch_size * gradient_accumulate_every) >= 16
        ), "your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above"

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.data_type = data_type
        if data_type == "image":
            self.ds = ImageData(
                folder,
                self.image_size,
                in_mem=in_memory,
                augment_horizontal_flip=augment_horizontal_flip,
                convert_image_to=convert_image_to,
                verbose=self.accelerator.is_main_process,
            )
        elif data_type == "vec":
            self.ds = VecData(folder)
        else:
            self.ds = LatentVarData(
                folder,
                scaling_factor,
                in_mem=in_memory,
                verbose=self.accelerator.is_main_process,
            )
        assert (
            len(self.ds) >= 100
        ), "you should have at least 100 images in your folder. at least 10k images recommended"
        real_workers = 0 if in_memory else min(cpu_count(), io_workers)
        prompt = (
            "only the main process loads the data"
            if in_memory
            else ("num of the worker processes for data loading: " + str(real_workers))
        )
        if self.accelerator.is_main_process:
            print("\n" + prompt, flush=True, end="\n\n")
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=real_workers,
        )
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )
            self.ema.to(self.device)
            if (data_type == "latent") and (pretrained_vae_dir is not None):
                self._vae_decoder = AutoencoderKL.from_pretrained(pretrained_vae_dir)
                self._vae_decoder.to(self.device)
                self._scaling_factor = scaling_factor

        self.results_folder = Path(results_folder)
        # self.results_folder.mkdir(exist_ok=True)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming. Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=test_batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx,
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite
        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"), map_location=device
        )
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            accum_loss, plot_buffer = 0.0, []

            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                accum_loss += total_loss
                pbar.set_description(f"loss: {total_loss:.4f}")
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(
                        self.step, self.save_and_sample_every
                    ):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(
                                self.num_samples, min(self.test_bsz, self.num_samples)
                            )
                            all_images_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(
                                        batch_size=n,
                                        clip_denoising=(self.data_type == "image"),
                                    ),
                                    batches,
                                )
                            )

                        all_images = torch.cat(all_images_list, dim=0)
                        snapshot_path = str(
                            self.results_folder / f"sample-{milestone}.png"
                        )
                        if self.data_type == "image":
                            utils.save_image(
                                all_images,
                                snapshot_path,
                                nrow=int(math.sqrt(self.num_samples)),
                            )
                        elif self.data_type == "vec":
                            samples = all_images.squeeze().cpu().numpy()
                            plt.figure()
                            plt.scatter(samples[:, 0], samples[:, 1], s=0.06)
                            plt.savefig(snapshot_path)
                        else:
                            latents = all_images / self._scaling_factor
                            with torch.no_grad():
                                mats = self._vae_decoder.decode(latents).sample
                            mats = (mats / 2 + 0.5).clamp(0, 1)
                            utils.save_image(
                                mats,
                                snapshot_path,
                                nrow=int(math.sqrt(self.num_samples)),
                            )

                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f"fid_score: {fid_score}")
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                        plot_buffer.append(
                            1.0 * accum_loss / self.save_and_sample_every
                        )
                        plt.figure()
                        plt.plot(
                            np.arange(1, 1 + len(plot_buffer))
                            * self.save_and_sample_every,
                            plot_buffer,
                        )
                        plt.ylabel("avg loss")
                        plt.xlabel("training steps")
                        plt.savefig(str(self.results_folder / "loss.png"))
                        accum_loss = 0.0

                pbar.update(1)

        accelerator.print("training complete")
