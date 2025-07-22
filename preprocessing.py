import numpy as np
import mne
from mne.filter import filter_data
import os

def preprocess_erf(file_path, output_folder, sfreq, h_freq):

    #load the erfs
    erf = np.load(file_path + "/erf.npy", allow_pickle=True)
    time_ = np.load(file_path + "/time.npy", allow_pickle=True)

    time = time_ - time_.min() #so that it's starting at 0 (dipole isn't understood otherwise)

    # check if the time is between 1000 and 1200, calculate mean and check if > 0 (if the main deflection is positive)
    time_mask = (time >= 1000.) & (time <= 1100.)
    max_peak = np.max(erf[time_mask])

    if max_peak < 0:
        reversed_flag = True
    else:
        reversed_flag = False

    # Flip the erf if needed
    if reversed_flag:
        erf_reversed = -erf
        flip_status = "flipped"  # Indicate that the ERF was flipped
    else:
        erf_reversed = erf
        flip_status = "not flipped"  # Indicate that the ERF was not flipped

    #butterworth filter if h_freq not None
    if h_freq != None:
        erf_filtered = filter_data(erf_reversed, sfreq, l_freq=0, h_freq=h_freq, method='iir', iir_params=None)
    else:
        erf_filtered = erf_reversed

    # stack the time and erf arrays (so understood by hnn as dipole)
    exp_dpl = np.column_stack((time, erf_filtered))
    
    # unique output name
    folder_name = os.path.basename(file_path)
    output_ = os.path.join(output_folder, f"{folder_name}_erf.txt")
    np.savetxt(output_, exp_dpl, fmt='%.18e', delimiter=' ')

    print(f"Processed {folder_name}. ERF was {flip_status}. Filtered at {h_freq} Saved to {output_}")


def batch_process(base_path, output_folder, sfreq, h_freq):
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        # verify that it is a directory and contains erf.npy and time.npy
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, "erf.npy")) and os.path.exists(os.path.join(folder_path, "time.npy")):
            preprocess_erf(folder_path, output_folder, sfreq, h_freq)
    