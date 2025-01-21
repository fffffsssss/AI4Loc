import ailoc.common

gt_array = ailoc.common.read_csv_array("ground_truth.csv")
pred_array = ailoc.common.read_csv_array("demo1_FD-DeepLoc_sim_data_normal_aberration_medium_SNR.csv")

metric_dict, paired_array = ailoc.common.pair_localizations(prediction=pred_array,
                                                            ground_truth=gt_array,
                                                            frame_num=2000,
                                                            fov_xy_nm=[0, 204800, 0, 204800],
                                                            print_info=True)
