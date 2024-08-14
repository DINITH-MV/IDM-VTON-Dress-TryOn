from flask import Flask, request, jsonify, send_file
import torch
from PIL import Image
import io
import os
import requests
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from torchvision import transforms
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

app = Flask(__name__)

# Initialize your models and variables here
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
unet = None
pipe = None
UNet_Encoder = None

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img

def initialize_pipeline():
    global unet, pipe, UNet_Encoder
    
    if pipe is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtype,
        )
        
        unet.requires_grad_(False)
        
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=dtype,
        )
        
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtype,
        )
     
        pipe_param = {
            'pretrained_model_name_or_path': model_id,
            'unet': unet,
            'torch_dtype': dtype,
            'vae': vae,
            'image_encoder': image_encoder,
            'feature_extractor': CLIPImageProcessor(),
        }
        
        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder
        pipe.unet_encoder.to(pipe.unet.device)

@app.route('/tryon', methods=['POST'])
def tryon():
    data = request.json
    human_image_url = data.get('human_image_url')
    garment_image_url = data.get('garment_image_url')
    garment_desc = data.get('garment_description', 'default description')
    
    # Load images
    human_img = load_image_from_url(human_image_url)
    garment_img = load_image_from_url(garment_image_url)
    
    # Ensure pipeline is initialized
    initialize_pipeline()
    
    # Resize and process images
    garment_img = garment_img.convert("RGB").resize((768, 1024))
    human_img = human_img.convert("RGB").resize((768, 1024))
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Example: Auto-generated mask
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
    mask = mask.resize((768, 1024))
    
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
    
    # Image Processing
    prompt = f"model is wearing {garment_desc}"
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            )
            
            garment_tensor = tensor_transform(garment_img).unsqueeze(0).to(device, dtype)
            generated_image = pipe(
                prompt_embeds=prompt_embeds.to(device, dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                num_inference_steps=30,
                strength=1.0,
                pose_img=None,
                text_embeds_cloth=None,
                cloth=garment_tensor.to(device, dtype),
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                guidance_scale=2.0,
                dtype=dtype,
                device=device,
            )[0]
    
    # Convert generated image to a byte stream and return as response
    img_byte_arr = io.BytesIO()
    generated_image[0].save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return send_file(io.BytesIO(img_byte_arr), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
