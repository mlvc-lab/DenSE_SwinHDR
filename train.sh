python3 -m torch.distributed.launch --nproc_per_node=5 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sdr2hdr.json --dist True
