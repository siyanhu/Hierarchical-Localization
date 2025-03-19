nohup bash nohup_recon_atrium.sh > nohup_recon_atrium.log 2>&1 &
nohup bash nohup_recon_concourse.sh > nohup_recon_concourse.log 2>&1 &
nohup bash nohup_recon.sh > nohup_recon.log 2>&1 &


ps aux | grep nohup


cp -r /media/siyanhu/T7/HKUST/scene_corridor/GX010041_42/images/* /media/siyanhu/T7/HKUST/scene_corridor/hloc_gopro/datasets/

ps aux | grep nohup_recon.sh | grep -v grep