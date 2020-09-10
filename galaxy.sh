python -m torch.distributed.launch --nproc_per_node=1 \
       glow.py --train \
               --dataset=galaxy \
               --n_levels=3 \
               --depth=16 \
               --width=256 \
               --batch_size=128 \
	       --n_epochs=1000
