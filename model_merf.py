'Create model based on initial net and params'

import numpy as np

def set_params(net, params, sim_shank=False, hypo=2):
    print(f"Setting parameters: {params}")

    weights_ampa_d1 = {'L2_basket': 0.001,
                       'L2_pyramidal': 0.005,
                       'L5_pyramidal': 0.002}

    net.add_bursty_drive('rydist1', 
                         tstart=775, tstart_std=100, tstop=2000, 
                         location='distal', burst_rate=50, burst_std=5, 
                         numspikes=2, spike_isi=10, n_drive_cells=1, cell_specific=False, 
                         weights_ampa=weights_ampa_d1, 
                         weights_nmda=None, 
                         synaptic_delays=0.1, 
                         space_constant=100.0, probability=1.0, event_seed=2, conn_seed=3)

    # Distal 2
    weights_ampa_d2 = {'L2_basket': params['evdist2_ampa_L2_basket'],
                       'L2_pyramidal': params['evdist2_ampa_L2_pyramidal'],
                       'L5_pyramidal': params['evdist2_ampa_L5_pyramidal']}
    weights_nmda_d2 = {'L2_basket': params['evdist2_nmda_L2_basket'],
                       'L2_pyramidal': params['evdist2_nmda_L2_pyramidal'],
                       'L5_pyramidal': params['evdist2_nmda_L5_pyramidal']}
    synaptic_delays_d2 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                      'L5_pyramidal': 0.1}

    net.add_evoked_drive('evdist2',
                         mu=params['evdist2_mu'],
                         sigma=params['evdist2_sigma'],
                         numspikes=1,
                         location='distal',
                         weights_ampa=weights_ampa_d2,
                         weights_nmda=weights_nmda_d2,
                         synaptic_delays=synaptic_delays_d2, event_seed=4)

    # Proximal 1
    weights_ampa_p1 = {'L2_basket': params['evprox1_ampa_L2_basket'],
                       'L2_pyramidal': params['evprox1_ampa_L2_pyramidal'],
                       'L5_basket': params['evprox1_ampa_L5_basket'],
                       'L5_pyramidal': params['evprox1_ampa_L5_pyramidal']}
    weights_nmda_p1 = {'L2_basket': params['evprox1_nmda_L2_basket'],
                       'L2_pyramidal': params['evprox1_nmda_L2_pyramidal'],
                       'L5_basket': params['evprox1_nmda_L5_basket'],
                       'L5_pyramidal': params['evprox1_nmda_L5_pyramidal']}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                        'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive('evprox1',
                         mu=params['evprox1_mu'],
                         sigma=params['evprox1_sigma'],
                         numspikes=1,
                         location='proximal',
                         weights_ampa=weights_ampa_p1,
                         weights_nmda=weights_nmda_p1,
                         synaptic_delays=synaptic_delays_prox)

    if hypo == 1:
        # Distal 3
        weights_ampa_d3 = {'L2_basket': params['evdist3_ampa_L2_basket'],
                           'L2_pyramidal': params['evdist3_ampa_L2_pyramidal'],
                           'L5_pyramidal': params['evdist3_ampa_L5_pyramidal']}
        weights_nmda_d3 = {'L2_basket': params['evdist3_nmda_L2_basket'],
                           'L2_pyramidal': params['evdist3_nmda_L2_pyramidal'],
                           'L5_pyramidal': params['evdist3_nmda_L5_pyramidal']}
        synaptic_delays_d3 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                          'L5_pyramidal': 0.1}
    
        net.add_evoked_drive('evdist3',
                             mu=params['evdist3_mu'],
                             sigma=params['evdist3_sigma'],
                             numspikes=1,
                             location='distal',
                             weights_ampa=weights_ampa_d3,
                             weights_nmda=weights_nmda_d3,
                             synaptic_delays=synaptic_delays_d3)
    if hypo == 2:
        # Proximal 2
        weights_ampa_p2 = {'L2_basket': params['evprox2_ampa_L2_basket'],
                           'L2_pyramidal': params['evprox2_ampa_L2_pyramidal'],
                           'L5_basket': params['evprox2_ampa_L5_basket'],
                           'L5_pyramidal': params['evprox2_ampa_L5_pyramidal']}
        weights_nmda_p2 = {'L2_basket': params['evprox2_nmda_L2_basket'],
                           'L2_pyramidal': params['evprox2_nmda_L2_pyramidal'],
                           'L5_basket': params['evprox2_nmda_L5_basket'],
                           'L5_pyramidal': params['evprox2_nmda_L5_pyramidal']}
        synaptic_delays_prox2 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                            'L5_basket': 1., 'L5_pyramidal': 1.}
        net.add_evoked_drive('evprox2',
                             mu=params['evprox2_mu'],
                             sigma=params['evprox2_sigma'],
                             numspikes=1,
                             location='proximal',
                             weights_ampa=weights_ampa_p2,
                             weights_nmda=weights_nmda_p2,
                             synaptic_delays=synaptic_delays_prox2)

    if sim_shank:
        # Simulate shank recordings - 0 is L5 cell bodies, cortex thickness about 2850um, 
        net.set_cell_positions(inplane_distance=30.)
        depths = list(range(-850,2100,int((2100--850)/11)))
        electrode_pos = [(135, 135, dep) for dep in depths]
        net.add_electrode_array('shank', electrode_pos)

    return net
