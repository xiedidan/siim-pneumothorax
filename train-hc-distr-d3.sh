python -m torch.distributed.launch --nproc_per_node=4 d3-hc-distr.py --size 768 --bs 2 --fold 4 --checkpoint 20190729-122311_512
