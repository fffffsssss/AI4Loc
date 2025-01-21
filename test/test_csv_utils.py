import torch
import ailoc.common

def test_csv():
    molecule_array = ailoc.common.read_csv_array('../test/ground_truth.csv')
    print('hh')

    molecule_array = ailoc.common.read_csv_array('../test/demo1_FD-DeepLoc_sim_data_normal_aberration_medium_SNR.csv')

if __name__ == '__main__':
    test_csv()