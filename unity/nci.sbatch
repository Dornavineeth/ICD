#!/bin/bash
#
# change mail-user to your email and run by command: sbatch abc.sbatch
#
#SBATCH --job-name=nci
#SBATCH --output=nci.txt                    # output file
#SBATCH -c 18                               # Number of Cores per Task
#SBATCH -p gypsum-rtx8000                   # Partition to submit to (serial_requeue), see here for all avaible resource: https://docs.unity.rc.umass.edu/technical/nodelist.html
#SBATCH -G 3                                # Number of GPUs
#SBATCH --mem=180G                          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -e nci.err                          # File to which STDERR will be written
#SBATCH --mail-user=vdorna@umass.edu        # Email to which notifications will be sent
#SBATCH --mail-type=ALL                     # Email for all types of Actions
#SBATCH -t 7-00:00:0                        # Job time limit 7-10:00:00 

module load miniconda/4.11.0
conda init bash
source activate /home/vdorna_umass_edu/.conda/envs/icd

nvidia-smi

cd /work/pi_hongyu_umass_edu/zhichao/GenRetrieval/ICD

bash scripts/longt5/train.sh

echo "Done"