#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name=lambdahat

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# The amount of memory in megabytes per node:
#SBATCH --mem=16384

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-1:00:00

#SBATCH --output=./outputs/slurm_logs/slurm_%A_%a.out
#SBATCH --error=./outputs/slurm_logs/slurm_%A_%a.err
#SBATCH --array=1-10

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load python/3.9.5


# Activate existing virtual environment
source /path/to/virtualenv/bin/activate

########## Script ###########
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" commands.txt)
echo "Executing command: $CMD"
eval $CMD

########## Script End ###########

# deactivate virtualenv
deactivate

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
