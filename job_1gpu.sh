srun --pty -t 0-04:00 --mem=128G -c 4 --gres=gpu:1 -p gpu_dia \
	     --mail-type=BEGIN,END,FAIL \
	     --mail-user=zongxin_yang@hms.harvard.edu \
		       /bin/bash
