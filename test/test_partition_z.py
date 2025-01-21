import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import time
import random


# Function to count z values in each interval
def count_intervals_1(roi_list,
                    intervals):
    counts = Counter()
    for roi in roi_list:
        for z in roi:
            for i in range(len(intervals) - 1):
                if intervals[i] <= z < intervals[i + 1]:
                    counts[(intervals[i], intervals[i + 1])] += 1
                    break
    return counts



# Function to select the best roi
def select_best_roi_1(roi_list,
                        current_counts,
                        required_counts,
                        intervals):
    best_roi = None
    best_score = sum((required_counts[interval] - current_counts[interval]) ** 2 for interval in required_counts)

    for roi in roi_list:
        new_counts = count_intervals_1([roi], intervals)
        temp_counts = current_counts + new_counts
        score = sum((required_counts[interval] - temp_counts[interval]) ** 2 for interval in required_counts)

        if score < best_score:
            best_roi = roi
            best_score = score

        # # break when any improvement exists
        # if best_score < sum((required_counts[interval] - current_counts[interval]) ** 2 for interval in required_counts):
        #     break

    return best_roi


def main_1():
    # each item in roi list contains a list of z with random lengths
    roi_list = []
    for i in range(50000):
        length = np.random.random_integers(1,5)
        z_list = []
        for j in range(length):
            z_list.append(np.clip(np.random.randn()*0.2*1000,a_min=-1000, a_max=1000))
        roi_list.append(z_list)


    # Define target interval counts
    bins = 20
    intervals = np.linspace(-1000, 1000, num=bins + 1)  # 10 intervals from -1 to 1
    required_counts = {interval: 100 for interval in zip(intervals[:-1], intervals[1:])}  # Example requirement

    # plot z distribution of original roi list
    z_list = []
    for i in range(len(roi_list)):
        z_list.extend(roi_list[i])
    plt.figure()
    plt.hist(np.array(z_list), bins=intervals)
    plt.show()

    # Create the new roi list
    new_roi_list = []
    current_counts = Counter()

    while True:
        best_roi = select_best_roi_1(roi_list,
                                           current_counts,
                                           required_counts,
                                           intervals)

        if best_roi is None:  # Check if no further improvement is possible
            break

        # # remove the best_sublist from the original_list
        # original_list.remove(best_sublist)
        new_counts = count_intervals_1([best_roi], intervals)
        current_counts.update(new_counts)
        new_roi_list.append(best_roi)

        if sum(current_counts.values()) > sum(required_counts.values()):
            break

    # plot the selected z list
    z_list_new = []
    for i in range(len(new_roi_list)):
        z_list_new.extend(new_roi_list[i])
    plt.figure()
    plt.hist(np.array(z_list_new), bins=intervals)
    plt.show()

    print(time.time() - t0)


# Function to select the best ROI
def select_best_roi_2(roi_list, current_counts, required_counts, intervals):
    # todo: feel like this function can be fully vectorized

    # Pre-compute required scores for ease of comparison
    current_array = np.array([current_counts[interval] for interval in required_counts])
    required_array = np.array([required_counts[interval] for interval in required_counts])

    best_roi = None
    best_roi_counts = Counter()
    best_score = np.sum((required_array - current_array) ** 2)

    # Determine the number of ROIs to check before selecting the best one
    check_num = 2000

    # Shuffle the roi_list to randomly check ROIs
    random.shuffle(roi_list)

    for i, roi in enumerate(roi_list):
        tmp_roi_counts = count_intervals([roi], intervals)
        tmp_roi_array = np.array([tmp_roi_counts[interval] for interval in required_counts])
        tmp_new_array = current_array + tmp_roi_array

        # Calculate score using vectorized operations
        tmp_score = np.sum((required_array - tmp_new_array) ** 2)

        if tmp_score < best_score:
            best_score = tmp_score
            best_roi = roi
            best_roi_counts = tmp_roi_counts

        # If we've checked enough ROIs, stop checking
        if i >= check_num:
            break

    return best_roi, best_roi_counts


def main_2():

    # Each item in roi_list contains a list of z with random lengths
    roi_list = [
        np.clip(
            np.random.randn(
                np.random.randint(1, 6)
            ) * 0.2 * 1000,
            a_min=-1000, a_max=1000-np.finfo(float).eps
        ).tolist()
        for _ in range(50000)
    ]

    # Define target interval counts
    bins = 20
    intervals = np.linspace(-1000, 1000, num=bins + 1)
    required_counts = {interval: 100 for interval in zip(intervals[:-1], intervals[1:])}

    # Plot Z distribution of original ROI list
    z_list = np.concatenate(roi_list)
    plt.figure()
    plt.hist(z_list, bins=intervals, alpha=0.7, label='Original')
    plt.title("Z Distribution of Original ROI List")
    plt.legend()
    plt.show()

    # Create the new ROI list
    new_roi_list = []
    new_roi_counts = Counter()

    while True:
        best_roi, best_roi_counts = select_best_roi_2(roi_list, new_roi_counts, required_counts, intervals)

        if best_roi is None:  # Check if no further improvement is possible
            break

        new_roi_counts.update(best_roi_counts)
        new_roi_list.append(best_roi)

        if sum(new_roi_counts.values()) >= sum(required_counts.values()):
            break

    # Plot the selected Z list
    z_list_new = np.concatenate(new_roi_list)
    plt.figure()
    plt.hist(z_list_new, bins=intervals, alpha=0.7, label='Selected')
    plt.title("Z Distribution of Selected ROI List")
    plt.legend()
    plt.show()


# Function to count z values in each interval using vectorization
def count_intervals(roi_list, intervals):
    counts = Counter()
    flat_list = np.concatenate(roi_list)  # Flatten the list of ROIs into a single array
    indices = np.digitize(flat_list, intervals) - 1  # Find the bin indices
    for idx in indices:
        if 0 <= idx < len(intervals) - 1:
            counts[(intervals[idx], intervals[idx + 1])] += 1
    return counts


def main_3():

    # Each item in roi_list contains a list of z with random lengths
    roi_list = [
        np.clip(
            np.random.randn(
                np.random.randint(1, 6)
            ) * 0.3 * 1000,
            a_min=-1000, a_max=1000-np.finfo(float).eps
        ).tolist()
        for _ in range(50000)
    ]

    # Define target interval counts
    bins = 20
    intervals = np.linspace(-1000, 1000, num=bins + 1)
    required_counts = {interval: 100 for interval in zip(intervals[:-1], intervals[1:])}
    # vectorized for better computation
    required_array = np.array([required_counts[interval] for interval in required_counts])

    # Plot Z distribution of original ROI list
    z_list = np.concatenate(roi_list)
    plt.figure()
    plt.hist(z_list, bins=intervals, alpha=0.7, label='Original')
    plt.title("Z Distribution of Original ROI List")
    plt.legend()
    plt.show()

    # calculate all roi's interval count for quick selection
    roi_counts_list = []
    for i, roi in enumerate(roi_list):
        tmp_counts = count_intervals([roi], intervals)
        tmp_counts_list = [tmp_counts[interval] for interval in required_counts]  # single roi interval counts
        tmp_counts_list.append(i)  # add the roi number at the end
        roi_counts_list.append(np.array(tmp_counts_list))   # transform the list into array
    roi_counts_array = np.array(roi_counts_list)   # collect all roi counts array for fast computation

    # Create the new ROI list
    new_roi_list = []
    new_roi_counts = Counter()

    while True:
        # calculate the current state and the score
        new_roi_array = np.array([new_roi_counts[interval] for interval in required_counts])
        best_score = np.sum((required_array - new_roi_array) ** 2)
        # calculate current roi list plus all candidate roi, and the score
        tmp_roi_array = new_roi_array + roi_counts_array[:, :-1]
        tmp_roi_score = np.sum((required_array - tmp_roi_array) ** 2, axis=1)

        # Check if any improvement,
        if any(tmp_roi_score < best_score):
            # get the best roi
            roi_indices = np.where(tmp_roi_score == tmp_roi_score.min())[0]
            best_roi_array_idx = roi_indices[np.random.randint(0, roi_indices.size)]
            best_roi_list_idx = roi_counts_array[best_roi_array_idx][-1]
            best_roi = roi_list[best_roi_list_idx]

            # update the new roi list and counts
            new_roi_counts.update(count_intervals([best_roi], intervals))
            new_roi_list.append(best_roi)

            # remove the selected roi from the roi counts array
            roi_counts_array = np.delete(roi_counts_array, best_roi_array_idx, axis=0)
        else:
            print('no better roi found')
            break

        if sum(new_roi_counts.values()) >= sum(required_counts.values()):
            print('found number satisfies')
            break

    # Plot the selected Z list
    z_list_new = np.concatenate(new_roi_list)
    plt.figure()
    plt.hist(z_list_new, bins=intervals, alpha=0.7, label='Selected')
    plt.title("Z Distribution of Selected ROI List")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    t0 = time.time()
    # main_1()
    # main_2()
    main_3()
    print(f"Execution Time: {time.time() - t0:.2f} seconds")
