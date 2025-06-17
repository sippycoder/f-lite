import glob
from typing import Any
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
# from torchvision.transforms.functional import exif_transpose
import torch
import logging
import time
import random
import io
import requests
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
import boto3
from tqdm import tqdm
from urllib3 import Retry
from dotenv import load_dotenv

load_dotenv()


def center_crop_arr(pil_image, image_size, max_ratio=1.0):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    crop_size = var_center_crop_size_fn(pil_image.size, image_size, max_ratio=max_ratio)

    crop_y = (pil_image.size[0] - crop_size[0]) // 2
    crop_x = (pil_image.size[1] - crop_size[1]) // 2
    return pil_image.crop(
        [crop_y, crop_x, crop_y + crop_size[0], crop_x + crop_size[1]]
    )


def generate_crop_size_list(image_size, max_ratio=2.0):
    assert max_ratio >= 1.0
    patch_size = 32     # patch size increments
    assert image_size % patch_size == 0
    min_wp, min_hp = image_size // patch_size, image_size // patch_size
    crop_size_list = []
    wp, hp = min_wp, min_hp
    while hp / wp <= max_ratio:
        crop_size_list.append((wp * patch_size, hp * patch_size))
        hp += 1
    wp, hp = min_wp + 1, min_hp
    while wp / hp <= max_ratio:
        crop_size_list.append((wp * patch_size, hp * patch_size))
        wp += 1
    return crop_size_list


def is_valid_crop_size(cw, ch, orig_w, orig_h):
    down_scale = max(cw / orig_w, ch / orig_h)
    return cw <= orig_w * down_scale and ch <= orig_h * down_scale


def var_center_crop_size_fn(orig_img_shape, image_size, max_ratio=2.0):
    """
    Dynamic cropping from Lumina-Image-2.0
    https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/imgproc.py#L39
    """
    w, h = orig_img_shape[:2]
    crop_size_list = generate_crop_size_list(
        image_size=image_size, 
        max_ratio=max_ratio
    )
    rem_percent = [
        min(cw / w, ch / h) / max(cw / w, ch / h) 
        if is_valid_crop_size(cw, ch, w, h) else 0 
        for cw, ch in crop_size_list
    ]
    crop_size = sorted(((x, y) for x, y in zip(rem_percent, crop_size_list) if x > 0), reverse=True)[0][1]
    return np.array(crop_size, dtype=np.int32)

class PolluxImageProcessing:
    def __init__(self, image_size, max_ratio):
        super().__init__()
        # Shared horizontal flip transform
        self.random_flip = transforms.RandomHorizontalFlip()

        # Separate ToTensor and Normalize for x
        self.normalize_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.image_size = image_size
        self.max_ratio = max_ratio

    def transform(self, x: Image) -> torch.Tensor:
        # 1. Apply center cropping
        x = center_crop_arr(x, self.image_size, self.max_ratio)

        # 2. Apply consistent random horizontal flip
        x = self.random_flip(x) # Flip the PIL image

        # 4. Apply ToTensor and Normalize separately
        x_normalized = self.normalize_transform(x)

        return x_normalized


class BaseDataset(Dataset):
    def __init__(
        self,
        collection_name: str,
        root_dir: str = "/fsx/metadata/training",
        root_dir_type: str = "parquet",
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.collection_name = collection_name
        self.data = None
        self.root_dir = root_dir
        self.root_dir_type = root_dir_type
        self.debug = debug
        
        logging.info("Data partition begins!")
        start_time = time.time()  # Record the start time
        
        data = []
        data_size = 0
        if self.root_dir_type == "parquet":
            logging.info(f"Loading data from local parquet files: {self.root_dir}, debug: {self.debug}")
            parquet_files = glob.glob(os.path.join(self.root_dir, self.collection_name, "*/*/*.parquet"))
            for file in tqdm(parquet_files, desc=f"Loading data"):
                df = pd.read_parquet(file)
                df = df[df["media_source"] != "laion"]
                data_size += len(df)
                data.append(df)
                
                # Note: used for debugging
                if self.debug and data_size > 10240:
                    break
        else:
            raise ValueError(f"Invalid Root Directory Type. Set root_dir_type to 'json' or 'parquet'")

        self.data = pd.concat(data, ignore_index=True)
        end_time = time.time()  # Record the end time
        # Calculate the duration in seconds
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        logging.info(
            f"Data Index retrieval from {self.root_dir}/{self.collection_name} completed in {int(minutes)} minutes and {seconds:.2f} seconds."
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class ImageDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        base_image_dir=None,
        image_column="media_path",
        caption_column="captions",
        resolution=512,
        center_crop=True,
        random_flip=False,
        keep_aspect_ratio=True,
        root_dir_type="parquet",
        base_url="s3://worldmodeldata-prod",
        debug=False,
    ) -> None:
        super().__init__(
            collection_name=data_path,
            root_dir=base_image_dir,
            root_dir_type=root_dir_type,
            debug=debug,
        )
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_processing = PolluxImageProcessing(resolution, max_ratio=1.0)
        self.retries = 3
        self.place_holder_image = Image.new("RGB", (resolution, resolution))
        
        # Create a session with connection pooling and retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=self.retries,
            backoff_factor=0.5,  # Exponential backoff
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"]
        )
        # Increase max connections per host
        adapter = HTTPAdapter(max_retries=retries, pool_connections=200, pool_maxsize=200)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.base_url = urlparse(base_url)
        if self.base_url.scheme == 'https':
            self.client = self.http_client
        elif self.base_url.scheme == 's3':
            self.client = self.s3_client
        elif self.base_url.scheme == 'dummy':
            self.client = self.dummy_client
        else:
            raise ValueError(f"Invalid scheme: {self.base_url.scheme}")

    def http_client(self, imageUrl: str) -> tuple[Image.Image, bool]:
        try:
            imageUrl = urlparse(imageUrl)._replace(netloc=self.base_url.netloc, scheme=self.base_url.scheme).geturl()

            head_response = self.session.head(imageUrl, timeout=1)
            if head_response.status_code != 200:
                raise requests.HTTPError(f"HEAD request failed with status code {head_response.status_code}")

            # Use session and increase timeout
            response = self.session.get(imageUrl, timeout=2, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            if random.random() > 0.9:
                self.place_holder_image = image  # frequently update the placeholder image
            return image, True # Signal success
        except (requests.RequestException, IOError) as e:
            status_code = getattr(locals().get('head_response'), 'status_code', 'N/A') # ensure head_response is accessed safely
            if isinstance(e, requests.Timeout):
                logging.debug(f"Timeout downloading image: {imageUrl}")
            elif isinstance(e, requests.HTTPError):
                logging.debug(f"HTTP error ({status_code}) for: {imageUrl}")
            elif isinstance(e, requests.ConnectionError):
                logging.debug(f"Connection error for: {imageUrl}")
            else:
                logging.debug(f"Error processing image {imageUrl}: {str(e)}")
            
            # Fall back to placeholder image
            return self.place_holder_image, False # Signal failure

    def s3_client(self, imageUrl: str) -> tuple[Image.Image, bool]:
        S3KEY = os.getenv("S3KEY")
        S3SECRET = os.getenv("S3SECRET")
        
        # Initialize s3 client here
        if not hasattr(self, 's3'):
            assert S3KEY is not None and S3SECRET is not None, "S3KEY and S3SECRET must be provided"
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=S3KEY,
                aws_secret_access_key=S3SECRET
            )
        try:
            imageUrl = urlparse(imageUrl)._replace(netloc=self.base_url.netloc, scheme=self.base_url.scheme)
            bucket_name = imageUrl.netloc
            object_key = imageUrl.path.lstrip('/') # Remove leading slash

            # Use the initialized self.s3 client
            response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
            image_data = response['Body'].read()
            
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            if random.random() > 0.9:
                self.place_holder_image = image  # frequently update the placeholder image
            return image, True # Signal success
        except Exception as e: 
            # Catching a broad exception. 
            # For production, you might want to catch more specific Boto3 exceptions like ClientError
            logging.warn(f"Error downloading image from S3 {imageUrl}: {str(e)}")
            return self.place_holder_image, False # Signal failure

        
    def dummy_client(self, imageUrl: str) -> tuple[Image.Image, bool]:
        return self.place_holder_image, True # Signal success

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # sample = self.data[idx]
        # for pd data
        sample = self.data.iloc[idx]  # Use iloc for row access in DataFrame
        return_sample = {}
        return_sample["_id"] = str(sample["source_id"])
        caption = sample[self.caption_column]
        if isinstance(caption, tuple) or isinstance(caption, list) or isinstance(caption, np.ndarray):
            caption = list(caption)
            if len(caption) == 0:
                logging.warning(f"The sample {sample} has no captions")
            caption = random.choice(caption) if len(caption) > 0 else ""

        if not isinstance(caption, str):
            logging.warning(f"Expected string but got {type(caption)}:{caption}")
            caption = ""
        return_sample["caption"] = caption
        
        image, success = self.client(sample[self.image_column])
        if success:
            return_sample[self.image_column] = self.image_processing.transform(image)
        else:
            # Fall back to placeholder image
            return_sample[self.image_column] = self.image_processing.transform(image) # image is placeholder here
            return_sample["_id"] = "-1"
            return_sample["caption"] = ""
        
        # Return image and metadata
        return (
            return_sample[self.image_column],
            [{
                "long_caption": return_sample["caption"],
            }]
        )
    
    def __del__(self):
        # Clean up the session when the dataset object is destroyed
        if hasattr(self, 'session'):
            self.session.close()
            
    def collate_fn(self, batch):
        return_batch = {}
        for k in batch[0].keys():
            items = [item[k] for item in batch]
            # Check if all items are tensors and have the same shape
            if all(isinstance(item, torch.Tensor) for item in items) and all(item.shape == items[0].shape for item in items):
                # Stack tensors if they all have the same shape
                return_batch[k] = torch.stack(items, dim=0)
            else:
                # Keep as list if not tensors or different shapes
                return_batch[k] = items
        return return_batch



if __name__ == "__main__":
    dataset = ImageDataset(
        data_path="train-bucket-1",
        base_image_dir="/fsx/metadata/training",
        root_dir_type="parquet",
        resolution=256,
        debug=True,
    )
    
    from PIL import Image

    for idx, data in enumerate(dataset):
        # print(idx, data)
        if idx > 100:
            break
    
        img_tensor = data[0]
        # Un-normalize from [-1, 1] to [0, 1], then scale to [0, 255]
        img_tensor = (img_tensor * 0.5 + 0.5) * 255
        # Permute from (C, H, W) to (H, W, C) and convert to uint8
        img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        # Create and save the image
        Image.fromarray(img_np).save(f"examples/test{idx}.png")
        print(f"Saved image {idx} with caption {data[1][0]['long_caption']}")