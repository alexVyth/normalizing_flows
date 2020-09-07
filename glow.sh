python -m torch.distributed.launch --nproc_per_node=1 \
       glow.py --train \
               --dataset=mnist \
               --data_dir=../ \
               --n_levels=3 \
               --depth=24 \
               --width=400 \
               --batch_size=100 \
	       --n_epochs=80
