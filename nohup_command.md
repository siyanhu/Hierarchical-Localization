nohup bash nohup_recon_atrium.sh > nohup_recon_atrium.log 2>&1 &
nohup bash nohup_recon_concourse.sh > nohup_recon_concourse.log 2>&1 &
nohup python nohup_scripts/run_recon.py > nohup_recon_full_hkust.log 2>&1 &


ps aux | grep nohup


cp -r /media/siyanhu/T7/HKUST/scene_corridor/GX010041_42/images/* /media/siyanhu/T7/HKUST/scene_corridor/hloc_gopro/datasets/

ps aux | grep nohup_recon.sh | grep -v grep


nohup python nohup_scripts/query_recon2.py > query_recon6.log 2>&1 &