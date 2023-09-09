import torch
import sys
sys.path.append('../../')
import os
import time

import ailoc.common


def deeploc_analyze():

    loc_model_path = '../../results/2023-09-01-11DeepLoc.pt'
    image_path = '../../datasets/match_data/1.tif'   # can be a tiff file path or a folder path
    save_path = '../../results/' + \
                os.path.split(loc_model_path)[-1].split('.')[0] + \
                '_'+os.path.split(image_path)[-1].split('.')[0]+'_predictions.csv'
    print(save_path)

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        deeploc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=deeploc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=0.01,
                                                     batch_size=16,
                                                     sub_fov_size=32,
                                                     over_cut=8,
                                                     num_workers=1)

    deeploc_analyzer.check_single_frame_output(frame_num=0)

    image_shape, fov_xy_nm, preds_rescale_array = deeploc_analyzer.divide_and_conquer()

    # read the ground truth and calculate metrics
    gt_array = ailoc.common.read_csv_array("../../datasets/match_data/activations.csv")

    metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_rescale_array,
                                                                ground_truth=gt_array,
                                                                frame_num=image_shape[0],
                                                                fov_xy_nm=fov_xy_nm,
                                                                print_info=True)
    # # write the paired localizations to csv file
    # save_paried_path = '../../results/'+os.path.split(save_path)[-1].split('.')[0]+'_paired.csv'
    # ailoc.common.write_csv_array(input_array=paired_array,
    #                              filename=save_paried_path,
    #                              write_mode='write paired localizations')


if __name__ == '__main__':
    deeploc_analyze()
