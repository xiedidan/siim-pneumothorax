python -m torch.distributed.launch --nproc_per_node=2 fp16-hc-distr.py --size 1024 --bs 1 --fold 4

