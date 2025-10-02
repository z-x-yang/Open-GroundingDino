#!/bin/bash
#SBATCH -c 5                              # Request one core
#SBATCH -t 1-00:00                        # Runtime in D-HH:MM format
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)
#SBATCH -p cpu
#SBATCH -o ./logs/%A_%a.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/%A_%a.out                 # File to which STDERR will be written, including job ID (%j)
#                                            You can change the filenames given with -o and -e to any filenames you'd like
#SBATCH --mail-type=FAIL                    # ALL email notification type
#SBATCH --mail-user=xuan_gong@hms.harvard.edu  # Email to which notifications will be sent
#SBATCH --array=0      # Run array for indexes. Should set to the length of the file list


IDX=$((SLURM_ARRAY_TASK_ID)) 

# sbatch preprocess_seg/mix_segpath.sh #60955
python preprocess_seg/mix_segpath.py