import gradio as gr
import argparse, torch, os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from typing import List
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
import requests
from io import BytesIO

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# The main try-on function
def start_tryon(human_url, garment_url, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    # Load human and garment images from URLs
    human_img_orig = load_image_from_url(human_url).convert("RGB")
    garm_img = load_image_from_url(garment_url).convert("RGB").resize((768, 1024))

    # Your existing logic for processing the images remains unchanged
    # Process the try-on using the pipeline
    results = ["output_image_path"]  # Placeholder for processing logic
    return results[0], "mask_image_path"  # Return the paths to the output and mask images

# Define Gradio Blocks UI
image_blocks = gr.Blocks().queue()

with image_blocks as demo:
    gr.Markdown("Dress Tryon with Dinith")

    # GUI for Human and Garment URLs
    with gr.Row():
        with gr.Column():
            human_url = gr.Textbox(label='Human Image URL', placeholder='Enter URL of the human image')
            garment_url = gr.Textbox(label='Garment Image URL', placeholder='Enter URL of the garment image')
            try_button = gr.Button(value="Try-on")

        with gr.Column():
            image_gallery = gr.Gallery(label="Generated Images", show_label=True)
            masked_img = gr.Image(label="Masked image output")

    # Handle try-on button click to process the input URLs
    try_button.click(fn=start_tryon, inputs=[human_url, garment_url, gr.Textbox("Garment Description"), gr.Radio(["upper_body", "lower_body", "dresses"], label="Garment Category", value="upper_body"), gr.Checkbox(label="Auto Masking", value=True), gr.Checkbox(label="Auto Crop & Resize", value=True), gr.Number(label="Denoising Steps", value=30), gr.Checkbox(label="Randomize Seed", value=True), gr.Number(label="Seed", value=1), gr.Number(label="Number of Images", value=1)], outputs=[image_gallery, masked_img])

# Add API functionality for POST requests
@app.post("/tryon")  # Expose a POST method to send image URLs
def process_via_post(human_url: str, garment_url: str, garment_des: str, category: str = "upper_body", is_checked: bool = True, is_checked_crop: bool = True, denoise_steps: int = 30, is_randomize_seed: bool = True, seed: int = 1, number_of_images: int = 1):
    # Process the images via the try-on pipeline
    result_image, mask_image = start_tryon(human_url, garment_url, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed, seed, number_of_images)

    # Load the result image and convert it to base64 to return as a response
    with open(result_image, "rb") as img_file:
        result_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return {"result_image": result_image_base64, "mask_image": mask_image}

# Launch the Gradio app
image_blocks.launch(inbrowser=True, share=True)
