## Ref: full lidar rgb3; Query: 2024 lidar rgb3
'''
    ref_dir = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/frames")
    queries_dir  = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/frames/2024")
    skip_query_seq_from_ref = ["2024"]
    outputs_ref = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/hloc")
'''

nohup python nohup_scripts/query_recon.py > query_recon.log 2>&1 &


## Ref: full lidar rgb5; Query: 2024 lidar rgb5
'''
    ref_dir = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames")
    queries_dir  = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames/2024")
    skip_query_seq_from_ref = ["2024"]
    outputs_ref = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/hloc")
'''

nohup python nohup_scripts/query_recon2.py > query_recon2.log 2>&1 &


## Ref: 2025 lidar rgb5 (independent db); Query: 2024 lidar rgb5
'''
    ref_dir = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames/2025")
    queries_dir  = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames/2024")
    skip_query_seq_from_ref = ["2024"]
    outputs_ref = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/hloc_rgb5_independent_2025")
'''

nohup python nohup_scripts/query_recon3.py > query_recon3.log 2>&1 &


## Ref: 2025 lidar rgb3 (independent db); Query: 2024 lidar rgb3
'''
    ref_dir = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/frames/2025")
    queries_dir  = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb3/frames/2024")
    skip_query_seq_from_ref = ["2024"]
    outputs_ref = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/hloc_rgb3_independent_2025")
'''

nohup python nohup_scripts/query_recon4.py > query_recon4.log 2>&1 &
