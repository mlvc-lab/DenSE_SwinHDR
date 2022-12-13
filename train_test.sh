CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python3 -m torch.distributed.launch --nproc_per_node=6 --master_port=1234 main_train_psnr_subin.py --opt options/swinir/train_swinir_sdr2hdr_test.json --dist True
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python3  main_train_psnr.py --opt options/swinir/train_swinir_sdr2hdr_test.json
