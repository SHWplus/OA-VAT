import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import *
# from DWPose.dwpose_utils import DWposeDetector
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline import Pose2ImagePipeline
from src.utils.util import import_filename, seed_everything
from einops import rearrange
import cv2
import pickle

warnings.filterwarnings("ignore")
from torch.autograd import Variable
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

class Net(nn.Module):
    def __init__(
        self,
        reid_net,
        ifr,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reid_net=reid_net
        self.ifr=ifr
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        feature_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        feature_embeds,_,_=self.reid_net(feature_embeds,feature_embeds,0,modal=1,seq_len=6)
        feature_embeds=self.ifr(feature_embeds)
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=feature_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=feature_embeds,
        ).sample
        return model_pred


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


def log_validation(
    reid_net,
    vae,
    net,
    scheduler,
    accelerator,
    width,
    height,
    generator,
):
    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider
    ifr=ori_net.ifr
    
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)

    # pose_detector = DWposeDetector()
    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_reid=transforms.Compose([
                            transforms.ToPILImage(),
                            RectScale(256, 128),
                            transforms.ToTensor(),
                            normalize
                        ])


    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_dir = args.ref_dir
    img_list = os.listdir(img_dir)
    img_list = [os.path.join(img_dir, x) for x in img_list]

    pose_dir = args.pose_dir
    pose_paths = os.listdir(pose_dir)
    pose_paths = [os.path.join(pose_dir, x) for x in pose_paths]

    inputs_list = []
    ref_image_list = []
    pose_image_list = []
    for i in range(len(img_list)):
        path_ref = img_list[i]
        img_ref = Image.open(path_ref)
        ref_image_pil = Image.open(path_ref).convert("RGB")
        ref_name=os.path.basename(path_ref)
        rgb_img = img_ref.resize((128, 256), Image.LANCZOS)
        #rgb_img = img_ref.resize((128, 256), Image.ANTIALIAS)
        rgb_img = np.array(rgb_img)
        reid_input=transform_reid(rgb_img).cuda()
        input = Variable(reid_input[None,...])
        inputs_list = []
        ref_image_list = []
        pose_image_list = []
        for j in range(len(pose_paths)):
            pose_path_ = pose_paths[j]
            # pose_image_pil, _ = pose_detector(cv2.imread(pose_path_))
            pose_image_pil = cv2.imread(pose_path_)
            pose_image_pil = Image.fromarray(pose_image_pil)
            
            inputs_list.append(input)
            ref_image_list.append(ref_image_pil)
            pose_image_list.append(pose_image_pil)


        inputs_list = torch.cat(inputs_list, dim=0)
        zeros_input = torch.zeros_like(inputs_list).cuda()
        inputs_list = torch.cat([zeros_input, inputs_list], dim=0)
        reid_output = reid_net(inputs_list,cam_label= torch.zeros((inputs_list.shape[0]), dtype=torch.long), view_label= torch.ones((inputs_list.shape[0]), dtype=torch.long))
        feature_embeds=ifr(reid_output).cuda()

        image = pipe(
            feature_embeds,
            ref_image_list,
            pose_image_list,
            width,
            height,
            20,
            3.5,
            batch_size = len(pose_paths),
            generator=generator,
        ).images
        w, h = 128, 256
        ref_image_pil = ref_image_list[0].resize((w, h))
        num = 1 + len(pose_paths) * 2
        canvas = Image.new("RGB", (w *num, h), "white")
        canvas.paste(ref_image_pil, (0, 0))

        for i in range(len(pose_paths)):
            image_i = image[i, :, 0].permute(1, 2, 0).cpu().numpy()
            res_image_pil = Image.fromarray((image_i * 255).astype(np.uint8))
            res_image_pil = res_image_pil.resize((w, h))
            pose_image_pil = pose_image_list[i].resize((w, h))
            canvas.paste(pose_image_pil, (i*w*2 +w, 0))
            canvas.paste(res_image_pil, (i*w*2 +2*w, 0))
        out = os.path.join(out_dir, ref_name)
        canvas.save(out)
        print(f"Saved to {out}")
        inputs_list = []
        ref_image_list = []
        pose_image_list = []
           
class IFR(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.num =20
        self.proj_motion = torch.nn.Linear(3840,  self.num *768)
        self.norm_motion = torch.nn.LayerNorm(768)

    def forward(self,encoder_hidden_states):
        encoder_hidden_states = self.proj_motion(encoder_hidden_states)
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b (n d) -> b n d', n=self.num)
        encoder_hidden_states = self.norm_motion(encoder_hidden_states)
        return encoder_hidden_states
    
def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    # load IFR model
    ifr=IFR().to(device="cuda")

    # load trans-reid model
    cfg_transreid=pickle.load(open('./cfg_transreid.pkl','rb'))
    from reidmodel.trainsreid import make_model
    reid_net = make_model(cfg_transreid, num_class=751, camera_num=0, view_num = 1)
    reid_net.load_param(args.ckpt_dir + "/transformer_20.pth")
    reid_net.to(device="cuda")
    reid_net.eval()

    # load pose guider 
    pose_guider = PoseGuider(
        conditioning_embedding_channels=320,
    ).to(device="cuda")

    # Load the diffusion models
    ckpt_dir = args.ckpt_dir
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"denoising_unet.pth"),
            map_location="cpu",
        ),
        strict=True,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"reference_unet.pth"),
            map_location="cpu",
        ),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"pose_guider.pth"),
            map_location="cpu",
        ),
        strict=True,
    )
    ifr.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, f"IFR.pth"),
            map_location="cpu", 
        ),
        strict=True,
    ) 
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reid_net,
        ifr,
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()


    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(cfg.seed)
    with torch.no_grad():
        sample_dicts = log_validation(
            reid_net,
            vae=vae,
            net=net,
            scheduler=val_noise_scheduler,
            accelerator=accelerator,
            width=cfg.data.train_height,
            height=cfg.data.train_width,
            generator = generator
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="pretrained")
    parser.add_argument("--pose_dir", type=str, default="standard_poses")
    parser.add_argument("--ref_dir", type=str, default="demo")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)

