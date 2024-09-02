#!/usr/bin/env python3

import glob
import os, sys
import shutil
import traceback

# Get the grandparent directory of the current script
grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the grandparent directory to sys.path
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)
    
import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

from saicinpainting.evaluation.masks.mask import SegmentationMask, propose_random_square_crop
from saicinpainting.evaluation.utils import load_yaml, SmallMode
from saicinpainting.training.data.masks import MixedMaskGenerator

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def resize_image(image, out_size, interpolation=cv2.INTER_CUBIC):
    """
    Resize an image using OpenCV.

    Args:
        image (numpy.ndarray or PIL.Image.Image): The input image to be resized.
        out_size (tuple): The desired output size as (width, height).
        interpolation (int, optional): The interpolation method to be used. 
            Default is cv2.INTER_CUBIC (cubic interpolation).

    Returns:
        PIL.Image.Image: The resized image as a PIL Image object.

    Raises:
        ValueError: If the input image is not a NumPy array or PIL Image.
        TypeError: If out_size is not a tuple.

    Example:
        image = Image.open('path/to/image.jpg')
        resized_image = resize_image(image, (800, 600))
    """
    if not isinstance(image, (np.ndarray, Image.Image)):
        raise ValueError("Input image must be a NumPy array or PIL Image.")
    
    if not isinstance(out_size, tuple):
        raise TypeError("out_size must be a tuple (width, height).")

    # Convert PIL Image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Resize the image using OpenCV
    resized_image_np = cv2.resize(image_np, out_size, interpolation=interpolation)

    # Convert the resized NumPy array back to PIL Image
    resized_image_pil = Image.fromarray(resized_image_np)

    return resized_image_pil


class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, indir, outdir, config):
    if config.generator_kind == 'segmentation':
        mask_generator = SegmentationMask(**config.mask_generator_kwargs)
    elif config.generator_kind == 'random':
        variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                              variants_n=variants_n)
    else:
        raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in tqdm(src_images):
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < config.cropping.out_min_size:
                handle_small_mode = SmallMode(config.cropping.handle_small_mode)
                if handle_small_mode == SmallMode.DROP:
                    continue
                elif handle_small_mode == SmallMode.UPSCALE:
                    factor = config.cropping.out_min_size / min(image.size)
                    out_size = (np.array(image.size) * factor).round().astype('uint32')
                    # image = image.resize(out_size, resample=Image.BICUBIC)
                    out_size = resize_image(image, tuple(out_size))
            else:
                factor = config.cropping.out_min_size / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                # image = image.resize(out_size, resample=Image.BICUBIC)
                out_size = resize_image(image, tuple(out_size))

            # generate and select masks
            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                else:
                    cur_image = image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((cur_image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                cur_image.save(cur_basename + '.png')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


def main(args):
    from nb_utils.file_dir_handling import list_files
    
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml(args.config)

    # in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))
    # print(f"args.indir: {args.indir}")
    if not os.path.exists(args.indir):
        print(f"Error .. Input directory not exists: {args.indir}")
        return 

    in_files = list_files(
        args.indir, 
        filter_ext=[".jpg", ".jpeg", ".png"], 
    )
    # print(f"in_files: {in_files}")
    print(f"Total input files: {len(in_files)}")

    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )
    print(f"Done!")

if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())
