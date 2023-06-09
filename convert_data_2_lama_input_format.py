# lama_format/convert_data_2_lama_input_format.py
"""
Purpose:
Prepare lama format data from default slicing result

it accepts mask file with "_mask" postfix on image filename.
So we need to remove "_orig" text from main image filename
"""
import os, sys
import shutil
from pathlib import Path
from loguru import logger
from psutil import cpu_count
from tqdm.contrib.concurrent import process_map, thread_map  # requires tqdm>=4.42.0
from functools import partial
from PIL import Image
from nb_utils.file_dir_handling import list_files

available_cpu = cpu_count(logical=False)
print(f"available_cpu: {available_cpu}")

def handle_single_image(image_file_rel_path, source_dir, out_dir, action="copy"):
    """
    copy or move original image and mask to output dir
    so object removal program will use that mask to remove that area object
    """
    p = Path(image_file_rel_path)
    image_file_abs_path = os.path.join(source_dir, image_file_rel_path)
    
    mask_path = image_file_abs_path.replace("_orig", "_mask")

    out_image_file_path = os.path.join(out_dir, image_file_rel_path)
    os.makedirs(os.path.dirname(out_image_file_path), exist_ok=True)
    out_mask_path = out_image_file_path.replace("_orig", "_mask")

    ## remove _orig from main filenames -- as per requirements of lama
    # ## update -- predict_v2.py -- no need to remove _orig prefix form it
    # out_image_file_path = out_image_file_path.replace("_orig", "")
    
    # ## check file extensions
    # if p.suffix.lower() in [".jpg", ".jpeg"]:
    #     # read image and save in png format
    #     img = Image.open(image_file_abs_path)
    #     out_image_file_path = Path(out_image_file_path).with_suffix(".png")
    #     img.save(out_image_file_path)

    #     ## read and save mask in png format
    #     img_mask = Image.open(mask_path)
    #     out_mask_path = Path(out_mask_path).with_suffix(".png")
    #     img_mask.save(out_mask_path)
    # else:
    if action == "move":
        dest = shutil.move(image_file_abs_path, out_image_file_path)
        dest_mask = shutil.move(mask_path, out_mask_path)
    else:
        dest = shutil.copy2(image_file_abs_path, out_image_file_path)
        dest_mask = shutil.copy2(mask_path, out_mask_path)

    return True

def convert_dataset_2_lama(source_dir, out_dir, action="copy", workers=-1):
    """
    Convert image mask pair to lama input format
    """
    if workers == -1:
        import psutil
        workers = psutil.cpu_count(logical=True)

    img_files = list_files(
        source_dir, filter_ext=[".jpg", ".jpeg", ".png"],
        return_relative_path=True
    )
    
    img_files_filtered = []
    for img_file in img_files:
        if "_orig." in img_file:
            img_files_filtered.append(img_file)

    logger.debug(f"Original_image_files count: {len(img_files_filtered)}")

    if workers == 1:
        for image_file_rel_path in img_files_filtered:
            handle_single_image(
                image_file_rel_path,
                source_dir=source_dir,
                out_dir=out_dir,
                action=action
            )
    else:
        worker = handle_single_image  # function to map
        kwargs = {
            'source_dir': source_dir,
            'out_dir': out_dir,
            'action': action,
        }
        jobs = img_files_filtered

        available_cpu = cpu_count(logical=False)
        all_result = thread_map(
            partial(worker, **kwargs), 
            jobs, max_workers=workers
        )

def main():
    from argparse import ArgumentParser

    # Initialize parser
    parser = ArgumentParser()

    parser.add_argument("-s", "--source_dir", type=str, required=True, help = "Source directory to read images.")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help = "Dest directory")
    parser.add_argument(
        "-a", "--action", type=str, required=False, 
        default="copy",
        help = "Action for file either copy or move"
    )
    parser.add_argument(
        "-w", "--workers", 
        type=int, default=-1, 
        help="If workers not equal to 1 multiprocessing will be used"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_dir):
        sys.exit(f"Source directory '{args.source_dir}' not exists.")

    source_dir = args.source_dir
    out_dir = args.out_dir
    action = args.action
    workers = args.workers

    # call  function
    convert_dataset_2_lama(source_dir, out_dir, action, workers=-1)

if __name__ == "__main__":
    main()