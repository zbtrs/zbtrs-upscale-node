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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "call"
    CATEGORY = "Dragon Upscaler"

    def call(self, image):
        image_url = self.upload_to_cos(image)
        if not image_url:
            raise Exception("Failed to upload image to COS.")

        response = self.upscale_image(image_url)
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

    def upscale_image(self, image_url):
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "input": {
                "image": image_url
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
