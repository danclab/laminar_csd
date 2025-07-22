'Utilities'

import os
import numpy as np
import nibabel as nib
import k3d
from scipy.spatial import KDTree, cKDTree

def get_roi_idx(subj_id, surf_dir, hemi, regions, surf):
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    roi_idx = []
    hemis = []
    if hemi is None:
        hemis.extend(['lh', 'rh'])
    else:
        hemis.append(hemi)
    for hemi in hemis:
        pial = nib.load(os.path.join(surf_dir, f'{hemi}.pial.gii'))

        annotation = os.path.join(fs_subject_dir, 'label', f'{hemi}.aparc.annot')
        label, ctab, names = nib.freesurfer.read_annot(annotation)

        name_indices = [names.index(region.encode()) for region in regions]
        orig_vts = np.where(np.isin(label, name_indices))[0]

        # Find the original vertices closest to the downsampled vertices
        kdtree = KDTree(pial.darrays[0].data[orig_vts, :])
        # Calculate the percentage of vertices retained
        dist, vert_idx = kdtree.query(surf.darrays[0].data, k=1)
        hemi_roi_idx = np.where(dist == 0)[0]
        roi_idx = np.union1d(roi_idx, hemi_roi_idx)
    return roi_idx.astype(int)

def convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord=None):
    """
    Convert coordinates from a subject's native surface space to the fsaverage surface space.

    This function maps a vertex coordinate from a subject's native combined pial surface
    to the corresponding vertex index in the fsaverage template space. If no coordinate
    is provided, it maps all vertices in the subject's downsampled pial surface to fsaverage.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like, optional
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be converted.
        If None, all downsampled pial vertices are mapped to fsaverage.

    Returns
    -------
    hemi : str or list of str
        The hemisphere(s) the vertex is found in ('lh' for left hemisphere, 'rh' for right
        hemisphere').
    fsave_v_idx : int or list of int
        Index or indices of the vertex on the fsaverage spherical surface that corresponds to the
        input coordinates.
    """
    fs_subjects_dir = os.getenv('SUBJECTS_DIR')
    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Load full-resolution and downsampled surfaces
    if subj_coord is None:
        # Load downsampled surface
        subj_ds = nib.load(os.path.join(subj_surf_dir, 'pial.ds.gii'))
        ds_vertices = subj_ds.darrays[0].data
    else:
        ds_vertices = np.array([subj_coord])

    # Load full-resolution pial surfaces
    subj_lh = nib.load(os.path.join(subj_surf_dir, 'lh.pial.gii'))
    lh_vertices = subj_lh.darrays[0].data
    subj_rh = nib.load(os.path.join(subj_surf_dir, 'rh.pial.gii'))
    rh_vertices = subj_rh.darrays[0].data

    # KDTree for finding the closest full-resolution vertex
    lh_kdtree = KDTree(lh_vertices)
    rh_kdtree = KDTree(rh_vertices)

    lh_dists, lh_pial_idx = lh_kdtree.query(ds_vertices, k=1)
    rh_dists, rh_pial_idx = rh_kdtree.query(ds_vertices, k=1)

    # Assign each vertex to the closest full-resolution vertex
    hemis = np.where(lh_dists < rh_dists, 'lh', 'rh')
    pial_vert_indices = np.where(lh_dists < rh_dists, lh_pial_idx, rh_pial_idx)

    # Load fsaverage spheres
    fsaverage_lh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'lh.sphere.reg')
    )
    fsaverage_rh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'rh.sphere.reg')
    )

    # Load subject registered sphere surfaces
    subj_lh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', 'lh.sphere.reg')
    )
    subj_rh_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subject_dir, 'surf', 'rh.sphere.reg')
    )

    # Precompute KDTree for fsaverage surfaces
    fs_lh_kdtree = KDTree(fsaverage_lh_sphere_vertices)
    fs_rh_kdtree = KDTree(fsaverage_rh_sphere_vertices)

    # Select appropriate subject sphere vertices
    subj_sphere_coords = np.array([
        subj_lh_sphere_vertices[idx] if hemi == 'lh' else subj_rh_sphere_vertices[idx]
        for hemi, idx in zip(hemis, pial_vert_indices)
    ])

    # Map to fsaverage **in batch** (much faster than looping)
    fsave_v_idx = np.array([
        fs_lh_kdtree.query(coord, k=1)[1] if hemi == 'lh' else fs_rh_kdtree.query(coord, k=1)[1]
        for hemi, coord in zip(hemis, subj_sphere_coords)
    ])

    # Return results
    if subj_coord is not None:
        return hemis[0], fsave_v_idx[0]
    return hemis.tolist(), fsave_v_idx.tolist()
    
def get_bigbrain_layer_boundaries(subj_id, subj_surf_dir, subj_coord=None):
    """
    Get the cortical layer boundaries based on the Big Brain atlas for a specified coordinate
    in the subject's downsampled combined space. If subj_coord is None, this function returns
    the 6 proportional layer boundaries for every vertex in the downsampled mesh.

    Parameters
    ----------
    subj_id : str
        The subject identifier for which the conversion is being performed.
    subj_surf_dir : str
        The path containing the laMEG-processed subject surfaces.
    subj_coord : array-like or None, optional
        The x, y, z coordinates on the subject's combined hemisphere pial surface to be
        converted. If None, all downsampled pial vertices are mapped. Defaults to None.

    Returns
    -------
    vert_bb_prop : np.ndarray
        A 6 x M array of proportional layer boundaries (rows = 6 layer boundaries,
        columns = vertices), where M is the number of vertices in subj_coord (if provided)
        or in the downsampled mesh (if subj_coord is None). Values range between 0 and 1,
        which must be scaled by the cortical thickness to get layer depths in millimeters.
    """
    # Convert subject coordinate(s) to fsaverage vertex index
    hemi, fsave_v_idx = convert_native_to_fsaverage(subj_id, subj_surf_dir, subj_coord)

    # Retrieve or compute the Big Brain proportional layer boundaries
    # big_brain_proportional_layer_boundaries() is assumed to return a dict:
    #    {'lh': <6 x N_lh array>, 'rh': <6 x N_rh array>}
    bb_prop = big_brain_proportional_layer_boundaries()

    # If we only have a single coordinate, hemi will be a string; otherwise, it is a list of hemis
    if isinstance(hemi, str):
        # Single coordinate: just index directly
        vert_bb_prop = bb_prop[hemi][:, fsave_v_idx]
    else:
        # Multiple coordinates: build a 6 x M array
        vert_bb_prop = np.zeros((6, len(hemi)))
        for i, (v_h, idx) in enumerate(zip(hemi, fsave_v_idx)):
            vert_bb_prop[:, i] = bb_prop[v_h][:, idx]

    return vert_bb_prop

import numpy as np

def extract_layer_csd(sm_csd, bb_lb_roi, roi_idx, layer_name):
    """
    Extracts average CSD across a specified cortical layer or merged layers.

    Parameters:
    - smooth_csd_emp_all: list of 2D arrays (depth x time/freq)
    - bb_lb_roi: list of normalized (0-1) layer boundaries per ROI
    - roi_idx: list of ROI indices
    - layer_name: string like 'L2_3', 'L5', 'L2_5'

    Returns:
    - layer_csd: NumPy array of average CSD for each ROI
    """

    # possible layer boundaries & how they map as indexes in bb_layer_bounds
    layer_boundaries = ['L1', 'L2_3', 'L4', 'L5', 'L6']
    layer_idx_map = {
        'L1': (99, 0),
        'L2': (0, 1),
        'L3': (1, 2),
        'L4': (2, 3),
        'L5': (3, 4),
        'L6': (4, 5)
    }
    if '_' in layer_name:
        parts = layer_name.split('_')
        start_idx = layer_idx_map[parts[0]][0]
        end_idx   = layer_idx_map['L' + parts[1]][1]
    else:
        start_idx, end_idx = layer_idx_map[layer_name]

    layer_csd = []
    for i in range(len(roi_idx)):
        boundaries = [1 + 499 * x for x in bb_lb_roi[i]] 
        if start_idx == 99: #if L1
            start = 0
        else:
            start = int(np.ceil(boundaries[start_idx]))
        end = int(np.floor(boundaries[end_idx]))
        layer_mean = np.nanmean(sm_csd[i][start:end], axis=0) #ensure that of non NaN values
        layer_csd.append(layer_mean)

    return np.array(layer_csd)


def compute_activity_over_time(ts_activity, woi_idx, roi_idx, method='mean'):
    """
    Compute summary statistics of layer CSD over time windows for each ROI

    Parameters:
    - ts_activity: 2D array/list time series per vertices (n_verts_roi x n_timepoints), can be 'all' as well
    - woi_idx: list of (start_idx, end_idx) time window index pairs
    - roi_idx: list of ROI indices (length = n_rois)
    - method: str, how activity is averaged within the time window (mean, max_abs, mean_abs)

    Returns:
    - m_ts: ndarray of shape (n_windows x n_rois)
    """
    m_ts = []

    for start_idx, end_idx in woi_idx:
        if method == 'mean':
            m_ts_ = [np.mean(ts_activity[i][start_idx:end_idx]) for i in range(len(roi_idx))]
        elif method == 'mean_abs':
            m_ts_ = [np.mean(np.abs(ts_activity[i][start_idx:end_idx])) for i in range(len(roi_idx))]
        elif method == 'max_abs':
            m_ts_ = [np.max(np.abs(ts_activity[i][start_idx:end_idx])) for i in range(len(roi_idx))]
            
        m_ts.append(m_ts_)

    return np.array(m_ts)
    

def find_clusters(faces, threshold_indices, n_hops=1):

    from lameg.surf import mesh_adjacency
    from scipy.sparse.csgraph import connected_components
    
    adjacency_matrix = mesh_adjacency(faces).tocsr()

    # Build N-hop adjacency matrix
    expanded_adj = adjacency_matrix.copy()
    power = adjacency_matrix.copy()

    for _ in range(1, n_hops):
        power = power @ adjacency_matrix  # Matrix multiplication for hop expansion
        expanded_adj = expanded_adj + power

    # Binarize: set all non-zero entries to 1
    expanded_adj.data[:] = 1.0
    expanded_adj.eliminate_zeros()

    # Extract subgraph of thresholded vertices
    subgraph = expanded_adj[threshold_indices, :][:, threshold_indices]

    # Find connected components
    n_components, labels = connected_components(csgraph=subgraph, directed=False, return_labels=True)

    # Group into clusters
    clusters = []
    for i in range(n_components):
        cluster_indices = np.where(labels == i)[0]
        clusters.append([threshold_indices[idx] for idx in cluster_indices])

    return clusters