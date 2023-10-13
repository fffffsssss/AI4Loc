import torch
import sys
sys.path.append('../../')
import os
import time
import imageio

import ailoc.common


def syncloc_analyze():

    loc_model_path = '../../results/2023-10-11-11SyncLoc.pt'
    image_path = '../../datasets/mismatch_data2/'   # can be a tiff file path or a folder path
    save_path = '../../results/' + \
                os.path.split(loc_model_path)[-1].split('.')[0] + \
                '_'+os.path.split(image_path)[-1].split('.')[0]+'_predictions.csv'
    print(save_path)

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        syncloc_model = torch.load(f)

    # # plot evaluation performance during the training
    # phase_record = ailoc.common.plot_syncloc_record(syncloc_model)
    # # save the phase learned during the training
    # imageio.mimsave('../../results/'+os.path.split(loc_model_path)[-1].split('.')[0]+'_phase.gif',
    #                 phase_record,
    #                 duration=200)

    syncloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=syncloc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=16,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)

    syncloc_analyzer.check_single_frame_output(frame_num=3)

    preds_array, preds_rescale_array = syncloc_analyzer.divide_and_conquer()

    # read the ground truth and calculate metrics
    gt_array = ailoc.common.read_csv_array("../../datasets/mismatch_data2/activations.csv")

    metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_array,
                                                                ground_truth=gt_array,
                                                                frame_num=syncloc_analyzer.tiff_dataset.end_frame_num,
                                                                fov_xy_nm=syncloc_analyzer.fov_xy_nm,
                                                                print_info=True)
    # # write the paired localizations to csv file
    # save_paried_path = '../../results/'+os.path.split(save_path)[-1].split('.')[0]+'_paired.csv'
    # ailoc.common.write_csv_array(input_array=paired_array,
    #                              filename=save_paried_path,
    #                              write_mode='write paired localizations')


if __name__ == '__main__':
    syncloc_analyze()
