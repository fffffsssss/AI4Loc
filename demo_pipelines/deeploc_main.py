import sys
import os
sys.path.append('../')

import ailoc.deeploc


def deeploc_main_learning():
    #todo
    pass


def deeploc_main_inference():
    loc_model_path = 'E:/projects/FS_work/AI-Loc/AI-Loc_project/test/reproduce_efig.4/2023-07-06-21DeepLoc.pt'
    image_path = 'E:/projects/FS_work/AI-Loc/AI-Loc_project/test/reproduce_efig.4/match_data/1.tif'
    if os.path.isfile(image_path):
        save_path = os.path.split(image_path)[-2] + '/' + \
                    os.path.split(loc_model_path)[-1].split('.')[0] + '_' + \
                    os.path.split(image_path)[-2].split('/')[-1] + '.csv'
    else:
        save_path = image_path + '/' + \
                    os.path.split(loc_model_path)[-1].split('.')[0] + '_' + \
                    os.path.split(image_path)[-1].split('/')[-1] + '.csv'

    time_block_gb = 1.0
    batch_size = 16
    sub_fov_size = 128
    over_cut = 8
    num_workers = 0
    fov_xy_start = [0, 0]

    ailoc.deeploc.app_inference(loc_model_path,
                                image_path,
                                save_path,
                                time_block_gb,
                                batch_size,
                                sub_fov_size,
                                over_cut,
                                num_workers,
                                fov_xy_start)


if __name__ == '__main__':
    # deeploc_main_learning()
    deeploc_main_inference()
    # ailoc.deeploc.app_inference_gui()
