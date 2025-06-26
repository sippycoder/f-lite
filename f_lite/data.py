import glob
from typing import Any
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
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
import ijson

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
boto3.set_stream_logger("boto3", level=logging.WARNING)
boto3.set_stream_logger("botocore", level=logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
Image.MAX_IMAGE_PIXELS = None

load_dotenv()


def center_crop_arr_simulator(orig_image_size, image_size, max_ratio=1.0):
    """
    Simulate the crop size calculation from center_crop_arr without creating any images.
    
    Args:
        orig_image_size: Tuple of (width, height) of the original image
        image_size: Target image size (minimum dimension)
        max_ratio: Maximum ratio of the dimensions
    
    Returns:
        Tuple of (crop_width, crop_height) that would be used for cropping
    """
    # Convert to (width, height) format to match PIL convention
    current_size = orig_image_size  # (w, h)
    
    # Step 1: Simulate repeated downsampling by 2x while min dimension >= 2 * image_size
    # while min(*current_size) >= 4 * image_size:
    #     current_size = tuple(x // 2 for x in current_size)
    
    # Step 2: Use var_center_crop_size_fn to get the final crop size
    crop_size = var_center_crop_size_fn(current_size, image_size, max_ratio=max_ratio)

    # # Step 3: Calculate scale factor to make minimum dimension equal to image_size
    # scale = max(crop_size[0] / current_size[0], crop_size[1] / current_size[1])
    # current_size = tuple(round(x * scale) for x in current_size)
    
    return crop_size


def center_crop_arr(pil_image, image_size, max_ratio=1.0):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """

    crop_size = var_center_crop_size_fn(pil_image.size, image_size, max_ratio=max_ratio)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.LANCZOS, reducing_gap=3.0
    )

    crop_y = (pil_image.size[0] - crop_size[0]) // 2
    crop_x = (pil_image.size[1] - crop_size[1]) // 2
    return pil_image.crop(
        [crop_y, crop_x, crop_y + crop_size[0], crop_x + crop_size[1]]
    )


# def generate_crop_size_list(image_size, max_ratio=2.0):
#     assert max_ratio >= 1.0
#     patch_size = 32     # patch size increments
#     assert image_size % patch_size == 0
#     min_wp, min_hp = image_size // patch_size, image_size // patch_size
#     crop_size_list = []
#     wp, hp = min_wp, min_hp
#     while hp / wp <= max_ratio:
#         crop_size_list.append((wp * patch_size, hp * patch_size))
#         hp += 1
#     wp, hp = min_wp + 1, min_hp
#     while wp / hp <= max_ratio:
#         crop_size_list.append((wp * patch_size, hp * patch_size))
#         wp += 1
#     return crop_size_list


def generate_crop_size_list(image_size, max_ratio=2):
    assert max_ratio >= 1
    patch_size = 16     # patch size increments
    assert image_size % patch_size == 0
    min_wp, min_hp = image_size // patch_size, image_size // patch_size
    max_wp, max_hp = image_size * max_ratio // patch_size, image_size * max_ratio // patch_size
    crop_size_list = []
    wp, hp = min_wp, max_hp
    while wp <= max_wp and hp >= min_hp:
        crop_size_list.append((round(wp * patch_size), round(hp * patch_size)))
        wp += 1
        hp -= 1
    return crop_size_list


def is_valid_crop_size(cw, ch, orig_w, orig_h, eps=1e-7):
    down_scale = max(cw / orig_w, ch / orig_h)
    return cw <= orig_w * down_scale + eps and ch <= orig_h * down_scale + eps


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
    return crop_size

class PolluxImageProcessing:
    def __init__(self, image_size, max_ratio):
        super().__init__()
        # Shared horizontal flip transform
        # self.random_flip = transforms.RandomHorizontalFlip()

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
        # x = self.random_flip(x) # DO NOT FLIP THE IMAGE - Important for learning text in images

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
            self.data = pd.concat(data, ignore_index=True)
        elif self.root_dir_type == "json":
            logging.info(f"Loading data from local parquet files: {self.root_dir}")

            file_path = os.path.join(self.root_dir, f"{self.collection_name}.json")
            data = []
            with open(file_path, "r") as file:
                for item in tqdm(ijson.items(file, "item"), desc=f"Loading data"):
                    data.append(item)
                    # Note: used for debugging
                    if self.debug and len(data) > 10000000:
                        break
            self.data = pd.DataFrame(data).reset_index()
        else:
            raise ValueError(f"Invalid Root Directory Type. Set root_dir_type to 'json' or 'parquet'")

        end_time = time.time()  # Record the end time
        # Calculate the duration in seconds
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        logging.info(
            f"Data Index retrieval from {self.root_dir}/{self.collection_name} completed in {int(minutes)} minutes and {seconds:.2f} seconds."
        )

    def __len__(self) -> int:
        return len(self.data) // 2048 * 2048

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
        self.image_processing = PolluxImageProcessing(resolution, max_ratio=1.0 if center_crop else 2.0)
        self.retries = 3

        crop_size_list = generate_crop_size_list(resolution, max_ratio=1.0 if center_crop else 2.0)
        self.place_holder_image = {
            (w, h): Image.new("RGB", (w, h))
            for w, h in crop_size_list
        }
        
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

    def setup_aspect_ratio_buckets(self, min_side, max_ratio):
        self.aspect_ratio_buckets = {}
        logging.info(f"Setting up aspect ratio buckets for {len(self.data)} images ...")
        time_start = time.time()

        # Create a cache for aspect ratio buckets
        w_h_array = self.data[["width", "height"]].to_numpy()
        aspect_ratio_bucket_cache = {}
        
        for idx, orig_image_shape in enumerate(w_h_array):
            orig_image_shape = tuple(orig_image_shape)
            if orig_image_shape not in aspect_ratio_bucket_cache:  # if not in cache, compute and add to cache
                # Get image resolution after var_center_crop_size_fn
                aspect = center_crop_arr_simulator(orig_image_shape, min_side, max_ratio)
                if aspect not in self.aspect_ratio_buckets:
                    self.aspect_ratio_buckets[aspect] = []
                aspect_ratio_bucket_cache[orig_image_shape] = aspect
            else:  # if in cache, get the aspect ratio from cache
                aspect = aspect_ratio_bucket_cache[orig_image_shape]
            self.aspect_ratio_buckets[aspect].append(idx)
        
        # Delete the cache
        del aspect_ratio_bucket_cache
        
        setup_time = time.time() - time_start
        logging.info(f"Created {len(self.aspect_ratio_buckets)} aspect ratio buckets with keys: {list(self.aspect_ratio_buckets.keys())} in {setup_time:.2f} seconds")

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
                bucket = center_crop_arr_simulator(
                    (image.width, image.height), 
                    self.image_processing.image_size, 
                    self.image_processing.max_ratio
                )
                self.place_holder_image[bucket] = image  # frequently update the placeholder image
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
            return None, False # Signal failure

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
                bucket = center_crop_arr_simulator(
                    (image.width, image.height), 
                    self.image_processing.image_size, 
                    self.image_processing.max_ratio
                )
                self.place_holder_image[bucket] = image  # frequently update the placeholder image
            return image, True # Signal success
        except Exception as e: 
            # Catching a broad exception. 
            # For production, you might want to catch more specific Boto3 exceptions like ClientError
            logging.warn(f"Error downloading image from S3 {imageUrl}: {str(e)}")
            return None, False # Signal failure

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
            expected_w, expected_h = center_crop_arr_simulator(
                (int(sample["width"]), int(sample["height"])), 
                self.image_processing.image_size, 
                self.image_processing.max_ratio
            )
            place_holder_image = self.place_holder_image[(expected_w, expected_h)]
            return_sample[self.image_column] = self.image_processing.transform(place_holder_image) # image is placeholder here
            return_sample["_id"] = "-1"
            return_sample["caption"] = ""
        
        # Return image and metadata
        return {
            "image": return_sample[self.image_column],
            "index": idx,
            "caption": return_sample["caption"],
            "media_source": sample["media_source"],
            "media_type": sample["media_type"],
        }
    
    def __del__(self):
        # Clean up the session when the dataset object is destroyed
        if hasattr(self, 'session'):
            self.session.close()

    def collate_fn(self, batch):
        """
        Custom collate_fn that ensures all image tensors have the same shape.

        If images in the incoming batch have different spatial resolutions (e.g. due to
        a rare bucket assignment mismatch) the minority-shape samples are replaced by
        randomly selected samples drawn from the majority-shape subset **together with
        all of their metadata**. This guarantees that, after the fix-up step, every
        image tensor in the batch can be safely stacked with `torch.stack`.
        """

        # ------------------------------------------------------------------
        # 1) Detect shape mismatch inside the batch
        # ------------------------------------------------------------------
        image_shapes = [sample["image"].shape for sample in batch]
        if len(set(image_shapes)) > 1:
            logging.info(f"Found {len(set(image_shapes))} different image shapes in the batch: {set(image_shapes)}")
            # There is at least one outlier – compute the majority shape.
            from collections import Counter

            shape_counter = Counter(image_shapes)
            # Select the most common shape (if there is a tie, random.choice breaks it)
            max_freq = max(shape_counter.values())
            majority_shapes = [s for s, c in shape_counter.items() if c == max_freq]
            majority_shape = random.choice(majority_shapes)

            # Pre-compute indices of samples that already have the majority shape
            majority_indices = [i for i, s in enumerate(image_shapes) if s == majority_shape]

            # Iterate through the batch and patch minority samples
            for idx, shape in enumerate(image_shapes):
                if shape != majority_shape:
                    logging.info(f"Patching minority sample {batch[idx]['index']} with shape {shape} with majority shape {majority_shape}")
                    # Randomly pick a donor sample with the majority shape and clone it
                    donor_idx = random.choice(majority_indices)
                    batch[idx] = batch[donor_idx]

            # Update image_shapes – now they should all be identical
            image_shapes = [sample["image"].shape for sample in batch]

            # Sanity check (avoid silent errors in the future)
            assert len(set(image_shapes)) == 1, "Failed to homogenise image shapes in collate_fn"

        # ------------------------------------------------------------------
        # 2) Standard collation – now every tensor is guaranteed to have the same shape
        # ------------------------------------------------------------------
        return_batch = {}
        for k in batch[0].keys():
            items = [item[k] for item in batch]
            if all(isinstance(item, torch.Tensor) for item in items):
                # All tensors now share the same shape → safe to stack
                return_batch[k] = torch.stack(items, dim=0)
            else:
                # Non-tensor data (e.g. caption strings) – keep as list
                return_batch[k] = items

        return return_batch



if __name__ == "__main__":
    os.makedirs("examples", exist_ok=True)
    collection_name = "train-bucket-4"
    os.makedirs(f"examples/{collection_name}", exist_ok=True)

    dataset = ImageDataset(
        data_path=collection_name,
        base_image_dir="/fsx/metadata/training",
        root_dir_type="parquet",
        resolution=256,
        center_crop=False,
        debug=True,
    )
    
    from PIL import Image

    for idx, data in enumerate(dataset):
        # print(idx, data)
        if idx > 100:
            break
    
        img_tensor = data["image"]
        # Un-normalize from [-1, 1] to [0, 1], then scale to [0, 255]
        img_tensor = (img_tensor * 0.5 + 0.5) * 255
        # Permute from (C, H, W) to (H, W, C) and convert to uint8
        img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        # Create and save the image
        Image.fromarray(img_np).save(f"examples/{collection_name}/test{idx}.png")
        print(f"Saved image {idx} with caption: \t\n{data['caption']}\n")