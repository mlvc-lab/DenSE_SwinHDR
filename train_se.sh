python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=1236 main_train_psnr.py --opt options/swinir/train_swinir_sdr2hdr_SE.json --dist True
