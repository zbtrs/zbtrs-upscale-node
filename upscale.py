import requests
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from io import BytesIO
import os
import time
import base64
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import json

COS_CONFIG_PATH = "/data1/zbtrs/ComfyUI/custom_nodes/ComfyUI-zbtrs-upscale/cos_config.json"

# Read COS configuration
with open(COS_CONFIG_PATH, "r") as file:
    cos_setting = json.load(file)

cos_config = CosConfig(Region=cos_setting['region'], SecretId=cos_setting['secret_id'], SecretKey=cos_setting['secret_key'])
cos_client = CosS3Client(cos_config)
cdn = 'aigc-cos.rabbitpre.com'
ROOT_API = "http://36.133.178.12:10009/predictions"

class DragonUpscaleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "(worst quality, low quality, normal quality:2) JuggernautNegative-neg"}),
                "scale_factor": (["2", "4", "6", "8", "10", "12", "14", "16"],),
                "dynamic": ("FLOAT", {"default": 6, "min": 0, "max": 10, "step": 1}),
                "resemblance": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 1.6, "step": 0.1}),
                "creativity": ("FLOAT", {"default": 0.35, "min": 0, "max": 1, "step": 0.01}),
                "tiling_width": ("FLOAT", {"default": 112, "min": 0, "max": 1000, "step": 1}),
                "tiling_height": ("FLOAT", {"default": 144, "min": 0, "max": 1000, "step": 1}),
                "num_inference_steps": ("FLOAT", {"default": 18, "min": 0, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call"
    CATEGORY = "Dragon Upscaler"

    def call(self, *args, **kwargs):
        image = kwargs.get('image', None)
        prompt = kwargs.get('prompt',"masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>")
        negative_prompt = kwargs.get('negative_prompt','(worst quality, low quality, normal quality:2) JuggernautNegative-neg')
        scale_factor = kwargs.get('scale_factor','2')
        dynamic = kwargs.get('dynamic',6)
        resemblance = kwargs.get('resemblance',0.6)
        creativity = kwargs.get('creativity',0.35)
        tiling_width = kwargs.get('tiling_width',112)
        tiling_height = kwargs.get('tiling_height',144)
        num_inference_steps = kwargs.get('num_inference_steps',18)
        
        image_url = self.upload_to_cos(image)
        if not image_url:
            raise Exception("Failed to upload image to COS.")

        response = self.upscale_image(image_url,prompt,negative_prompt,scale_factor,dynamic,resemblance,creativity,tiling_width,tiling_height,num_inference_steps)
        if response.status_code == 200:
            base64_data = response.json().get("output")[0]
            return self.base64_to_image(base64_data)
        else:
            raise Exception("Upscale service failed with status code: " + str(response.status_code))

    def upload_to_cos(self, image):
        buffer = BytesIO()
        image = ToPILImage()(image.squeeze(0).permute(2,0,1))
        image.save(buffer, format='PNG')
        buffer.seek(0)
        object_key = f"images/{int(time.time())}.png"

        response = cos_client.put_object(
            Bucket=cos_setting['bucket'],
            Body=buffer.getvalue(),
            Key=object_key
        )

        if response['ETag']:
            cos_url = f"https://{cdn}/{object_key}"
            return cos_url
        else:
            return None

    def upscale_image(self, image_url,prompt,negative_prompt,scale_factor,dynamic,resemblance,creativity,tiling_width,tiling_height,num_inference_steps):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "input": {
                "image": image_url,
                "prompt": prompt,
                "negative_prompt":negative_prompt,
                "scale_factor":scale_factor,
                "dynamic":dynamic,
                "resemblance":resemblance,
                "creativity":creativity,
                "tiling_width":tiling_width,
                "tiling_height":tiling_height,
                "num_inference_steps":num_inference_steps
            }
        }
        response = requests.post(ROOT_API, headers=headers, json=data)
        return response

    def base64_to_image(self, base64_data):
        _,context=base64_data.split(",")  
        image_data = base64.b64decode(context)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGBA")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        return (image_tensor,)
