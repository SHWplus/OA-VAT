from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.biodrone_path = '/home/gao/ORTrack/data/biodrone'
    settings.davis_dir = ''
    settings.dtb70_path = '/home/gao/ORTrack/data/dtb70'
    settings.got10k_lmdb_path = '/home/gao/ORTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/gao/ORTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/gao/ORTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/gao/ORTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/gao/ORTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/gao/ORTrack/data/lasot'
    settings.network_path = '/home/gao/ORTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/gao/ORTrack/data/nfs'
    settings.otb_path = '/home/gao/ORTrack/data/otb'
    settings.prj_dir = '/home/gao/ORTrack'
    settings.result_plot_path = '/home/gao/ORTrack/output/test/result_plots'
    settings.results_path = '/home/gao/ORTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/gao/ORTrack/output'
    settings.segmentation_path = '/home/gao/ORTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/gao/ORTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/gao/ORTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/gao/ORTrack/data/trackingnet'
    settings.uav123_path = '/home/gao/ORTrack/data/uav123'
    settings.uav_path = '/home/gao/ORTrack/data/uav'
    settings.uavdt_path = '/home/gao/ORTrack/data/uavdt'
    settings.visdrone2018_path = '/home/gao/ORTrack/data/visdrone2018'
    settings.vot18_path = '/home/gao/ORTrack/data/vot2018'
    settings.vot22_path = '/home/gao/ORTrack/data/vot2022'
    settings.vot_path = '/home/gao/ORTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

