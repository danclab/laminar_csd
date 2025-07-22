## Use MPI backend to write and quickly store the MPI

import os
import numpy as np
import nibabel as nib
from lameg.invert import load_source_time_series
from utils import get_bigbrain_layer_boundaries, get_roi_idx, compute_activity_over_time, extract_layer_csd
from mpi4py import MPI #use MPI instead of joblib as it cannot write in parallel
import h5py  #to write directly and avoid ram problems

comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # process ID: 0..N-1
size = comm.Get_size()      # total number of processes

#As dependencies: the matrix MU for source reconstruction, the base_fname of the single trial dataset

subj_id='sub-001'
ses_id = 'ses-01'
epoch='motor'
c_idx=3
subj_dir=os.path.join('/home/common/bonaiuto/cued_action_meg/derivatives/processed',subj_id)
subj_dir_sss=os.path.join('/home/common/bonaiuto/cued_action_meg/derivatives/processed_sss',subj_id)
subj_surf_dir=os.path.join(subj_dir,'surf')
multilayer_mesh_fname = os.path.join(subj_surf_dir, 'multilayer.11.ds.link_vector.fixed.gii')
pial_mesh_fname = os.path.join(subj_surf_dir,'pial.ds.link_vector.fixed.gii')
out_dir=os.path.join('./data', subj_id, ses_id, f'{subj_id}_{ses_id}_c{c_idx}_{epoch}_model_inv')
out_dir_chunks = os.path.join(out_dir, 'csd_chunks_signif')
os.makedirs(out_dir_chunks, exist_ok=True)

os.environ['SUBJECTS_DIR']='/home/common/bonaiuto/cued_action_meg/derivatives/processed/fs/'

#define the output file_path:
h5_filename_path = f'{out_dir_chunks}/group_data_st.h5'

# Compute the number of vertices per layer
mesh = nib.load(multilayer_mesh_fname)
pial_mesh = nib.load(pial_mesh_fname)
n_layers = 11
verts_per_surf = int(mesh.darrays[0].data.shape[0]/n_layers)

MU_fname = os.path.join(out_dir, 'MU.npy') 
MU = np.load(MU_fname, allow_pickle = True) 

base_fname_t=os.path.join(subj_dir_sss, ses_id, f'spm/pcspm_converted_autoreject-{subj_id}-{ses_id}-{epoch}-epo.mat')

roi_idx = get_roi_idx(subj_id, subj_surf_dir, 'lh', ['precentral','paracentral','postcentral'], pial_mesh)

bb_layer_bound_fname = os.path.join(out_dir, 'bb_layer_bound.npy') 
bb_layer_bound = np.load(bb_layer_bound_fname, allow_pickle = True) 
bb_lb_roi = bb_layer_bound[:,roi_idx]
bb_lb_roi = bb_lb_roi.T #just to get the vertices in first

layer_ts, time, _ = load_source_time_series(
        base_fname_t,
        mu_matrix=MU, #we base ourselves on the inversion matrix from averaged data
        vertices=5195
    )

wd_size = 0.05  # 50 ms
step_size = 0.05  # for non-overlapping windows; this reduce for overlap
start_time = time[300] #start_time = time[0] 
end_time = time[900] #end_time = time[-1]

#woi = [(-0.5, -0.1), (-0.1, 0), (0, 0.12), (0.12, 0.3)]
baseline_woi_idx = [(0,300)] 
woi = []
t = start_time
while t + wd_size <= end_time:
    woi.append((t, t + wd_size))
    t += step_size

woi_idx = [(
    (np.abs(time - start)).argmin(),
    (np.abs(time - end)).argmin()
) for start, end in woi]

'''
Process vertices like mean data in pipeline_05
'''
def process_vertex(vertex, bin_size=50):
    layer_verts = [l * int(verts_per_surf) + vertex for l in range(n_layers)]
    layer_coords = mesh.darrays[0].data[layer_verts, :]
    thickness = np.linalg.norm(layer_coords[0, :] - layer_coords[-1, :])
    vert_idx = np.where(roi_idx == vertex)[0][0] # vert idx in roi

    layer_ts, time, _ = load_source_time_series(
        base_fname_t,
        mu_matrix=MU, #we base ourselves on the inversion matrix from averaged data
        vertices=layer_verts
    )

    n_trials = layer_ts.shape[-1]
    n_woi = len(woi_idx)
    n_bins = n_trials // bin_size

    mcsd_L5_all = np.zeros((n_bins, n_woi, 1))
    mcsd_L2_3_all = np.zeros((n_bins, n_woi, 1))
    m_pial_ts_all = np.zeros((n_bins, n_woi, 1))
    
    #bin trials to reduce computational load (will only slighlty reduce the statistical power)
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size

        binned_ts = np.mean(layer_ts[:, :, start:end], axis=2)

        _, sm_csd = compute_csd(
            binned_ts,
            thickness,
            sfreq=s_rate,
            smoothing='cubic'
        )

        csd_L5 = extract_layer_csd([sm_csd], [bb_lb_roi[vert_idx]], [vertex], 'L5')
        csd_L2_3 = extract_layer_csd([sm_csd], [bb_lb_roi[vert_idx]], [vertex], 'L2_3')

        mcsd_L5 = compute_activity_over_time(csd_L5, woi_idx, [vertex], method='mean_abs')
        mcsd_L2_3 = compute_activity_over_time(csd_L2_3, woi_idx, [vertex], method='mean_abs')
        m_pial_ts = compute_activity_over_time([binned_ts[0]], woi_idx, [vertex], method='mean_abs')

        mcsd_L5_all[b] = np.abs(mcsd_L5)
        mcsd_L2_3_all[b] = np.abs(mcsd_L2_3)
        m_pial_ts_all[b] = np.abs(m_pial_ts)

    return mcsd_L5_all, mcsd_L2_3_all, m_pial_ts_all

bin_size = 30
n_woi = len(woi_idx)
n_trials = layer_ts.shape[-1]
n_bins = n_trials // bin_size
n_vertices = len(roi_idx)

# How many vertices for this process?
vertices_per_rank = np.array_split(roi_idx, size)
my_vertices = vertices_per_rank[rank]

# Open HDF5 with parallel mode
with h5py.File(h5_filename_path, 'w', driver='mpio', comm=comm) as h5f:
    if rank == 0: #creates only once
        h5f.create_dataset('v_mcsd_L5', shape=(n_vertices, n_bins, n_woi, 1), dtype='float32')
        h5f.create_dataset('v_mcsd_L2_3', shape=(n_vertices, n_bins, n_woi, 1), dtype='float32')
        h5f.create_dataset('v_m_pial_ts', shape=(n_vertices, n_bins, n_woi, 1), dtype='float32')
        h5f.create_dataset('status', shape=(n_vertices,), dtype='uint8')
comm.Barrier()  # sync all processes

# Now each process opens the file in append mode and writes its chunk
with h5py.File(h5_filename_path, 'r+', driver='mpio', comm=comm) as h5f:
    for i, vert in enumerate(my_vertices):
        idx = np.where(roi_idx == vert)[0][0]  # global index in roi_idx
        
        if h5f['status'][idx] == 1:
            print(f"Rank {rank}: Vertex {vert} already processed, skipping.")
            continue
    
        # run the process function 
        mcsd_L5_all, mcsd_L2_3_all, m_pial_ts_all = process_vertex(vertex=vert, bin_size=bin_size)
        
        # write results
        h5f['v_mcsd_L5'][idx, ...] = mcsd_L5_all
        h5f['v_mcsd_L2_3'][idx, ...] = mcsd_L2_3_all
        h5f['v_m_pial_ts'][idx, ...] = m_pial_ts_all
        h5f['status'][idx] = 1
        print(f"Rank {rank}: Wrote vertex {vert} (index {idx})")

comm.Barrier()
if rank == 0:
    print("done!")