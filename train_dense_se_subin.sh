cd /root/subin/DenSE_Swin_SDR2HDR
pip install -r requirement.txt
apt-get -y update
apt-get -y install libgl1-mesa-glx
apt-get -y install libglib2.0-0
pip install timm
python main_train_psnr_subin.py --opt options/swinir/train_swinir_sdr2hdr_dense_se_subin.json