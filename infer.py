import argparse
import logging
import torch
import numpy as np
import torchvision.transforms as transforms
from leaf import LeafPipeline, AutoencoderKL, LatentEncoder, UNetModel
from diffusers import DDIMScheduler
from PIL import Image
from leaf import LeafPipeline, LeafOutput
import os
from leaf.fusion import GroupFusion   # 新增
def parse_args():
    parser = argparse.ArgumentParser(description="LeafPipeline inference script with argparse.")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to finetuned LeafPipeline."
    )
    parser.add_argument(
        "--input_image", 
        type=str, 
        required=True, 
        help="Path to input image."
    )
    parser.add_argument(
        "--output_image", 
        type=str, 
        default="./mask.png", 
        help="Path to save prediction mask."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to run on, e.g., 'cuda' or 'cpu'."
    )

    return parser.parse_args()

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="'%(asctime)s - %(levelname)s -%(filename)s - %(funcName)s >> %(message)s'",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def main():
    args = parse_args()
    setup_logger()

    logging.info(f"Using device: {args.device}")
    device = torch.device(args.device)
    
    #logging.info(f"Loading pipeline from {args.pretrained_path} ...")
    logging.info(f"Loading pipeline from {args.model_path} ...")
    device = torch.device(args.device)

    # 1) UNet（优先用 EMA）
    unet = UNetModel.from_pretrained("./outputqata_1/QATA-leaf-1/checkpoint/step-274000/unet")

    # 2) latent encoder（训练时单独保存了）
    latent_encoder = LatentEncoder.from_pretrained("./outputqata_1/QATA-leaf-1/checkpoint/step-274000/latent_encoder")

    # 3) VAE（训练阶段从 assets 取的，这里也一样）
    vae = AutoencoderKL.from_pretrained("../assets/vae")

    # 4) 调度器（与 train.py 保持一致：1000 步，beta_start/end，prediction_type）
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0155, prediction_type="sample"
    )
    # 5) 分组融合模块（和 train.py 的 save_model_hook 对应）
    fusion = GroupFusion.from_pretrained("./outputqata_1/QATA-leaf-1/checkpoint/step-274000/fusion")
    fusion = fusion.half().to(device)          # 统一成 float16
    # 6) 组装成 LeafPipeline
    # 构建 pipeline
    pipeline = LeafPipeline(
        vae=vae,
        unet=unet,
        latent_encoder=latent_encoder,
        scheduler=scheduler,
        fusion=fusion,  # 新增
    ).to(device)

    pipeline.set_progress_bar_config(disable=False)
    pipeline.unet.eval()
    pipeline.vae.eval()
    logging.info("Pipeline loaded and set to eval mode.")

    # Load input image
    logging.info(f"Loading input image: {args.input_image}")
    input_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    rgb = Image.open(args.input_image).convert("RGB")
    rgb = input_transforms(rgb).unsqueeze(0).to(device)
    logging.info(f"Input image transformed. Shape: {rgb.shape}")

    # Run inference
    logging.info("Running inference ...")
    with torch.autocast(device.type):
        output: LeafOutput = pipeline(
            rgb,
            num_inference_steps=1,
            show_progress_bar=False,
        )
    logging.info("Inference completed.")

    # Process mask
    logging.info("Processing mask ...")
    mask = torch.where(
        torch.mean(output.mask_pred, dim=1, keepdim=True) > 0.5, 
        1, 
        0
    )
    mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0)
    mask_arr = (mask.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(mask_arr)

    # Save result
    img.save(args.output_image)
    logging.info(f"Prediction mask saved to {args.output_image}")


if __name__ == '__main__':
    main()
