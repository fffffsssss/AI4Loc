import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import tifffile as tiff

import cv2 as cv
import numpy as np
from pathlib import Path
import natsort


def extract_and_save_rois(input_dir, output_file, roi):
    """
    Traverse the directory to get all .tif files, extract the specified ROI from each file,
    and save these ROIs into one file.

    Args:
    - input_dir: Directory containing the original .tif files.
    - output_file: Path to save the concatenated ROIs.
    - roi: Tuple (row_start, row_end, col_start, col_end) specifying the ROI.
    """
    # i = 0
    with tiff.TiffWriter(output_file, append=True, bigtiff=True) as tif:
        for file_path in natsort.natsorted(Path(input_dir).glob('*.tif')):
            print(f"Processing {file_path}")
            data = tiff.imread(file_path, key=slice(0, None))
            if len(data.shape) == 2:
                data = data[np.newaxis, ...]
            subvolume = data[:, roi[0]:roi[1], roi[2]:roi[3]]  # Extract ROI
            for frame in range(subvolume.shape[0]):
                tif.write(subvolume[frame], contiguous=True)

            # i += 1
            # if i >= 2:
            #     break
    print(f"Saved concatenated ROIs to {output_file}")


def test_read_rois(output_file):
    """
    Test reading the saved ROIs from the output file.

    Args:
    - output_file: Path to the saved concatenated ROIs file.
    """

    with tiff.TiffFile(output_file, is_ome=False) as tif:
        total_shape = tif.series[0].shape
        print(f"Total shape: {total_shape}")


if __name__ == '__main__':
    # # mitochondria
    # input_directory = 'V:/20220423/TOM20_cos7_Invitrogen_zhou_protocol_liu_lab_15ms_1800mw_1608_2'
    # # input_directory = 'V:/20220414/Tetrapod/Tetrapod_Crismo_beads_6um_100nm_1608_200mw_3'
    # output_file = 'V:/20220423/2_crop_FS/crop_roi.tif'
    # # output_file = 'V:/20220423/2_crop_FS/crop_beads.ome.tif'
    # roi = (275, 639, 470, 790)  # Define the ROI (row_start, row_end, col_start, col_end)

    # neuron
    input_directory = 'T:/20220923/Spectrin_Neuron_Day24_tetrapod_objective_-870nm_15ms_1800mw_1608_1'
    # input_directory = 'T:/20220923/Tetrapod/Tetrapod_Crimson_beads_4um_100nm_1608_1'
    output_file = 'T:/20220923/-870_crop_FS/neuron_crop.tif'
    # output_file = 'T:/20220923/-870_crop_FS/beads_crop.tif'
    roi = (1030, 1030+512, 916, 916+512)  # Define the ROI (row_start, row_end, col_start, col_end)

    extract_and_save_rois(input_directory, output_file, roi)

    test_read_rois(output_file)
