#!/bin/bash
#PBS -N batch_job_knn
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:gpu_cap=cuda80:gpu_mem=20gb:mem=20gb:scratch_ssd=20gb:cluster=galdor
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m ae

# -j oe ... standard error stream of the job will be merged with the standard output stream
# -m ae ...  mail is sent when the job aborts or terminates

# Source: https://github.com/roman-janik/diploma_thesis_program/blob/main/ner/train_ner_model.sh

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            # folder with datasets
RESPATH=$HOMEPATH/program/results/      # store results in this folder
HOSTNAME=$(hostname -f)                 # hostname of local machine

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             %s\n" "$PBS_JOBID"
printf "JOB NAME:           %s\n" "$PBS_JOBNAME"
printf "JOB SERVER NODE:    %s\n" "$HOSTNAME"
printf "START TIME:         %s\n" "$(date +%Y-%m-%d-%H-%M)"
#printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"

start_time=$(date +%s)

cd "$SCRATCHDIR" || exit 2

# clean the SCRATCH directory
clean_scratch

# Clone the repository
printf "Cloning the repository ...\n"
cp "$HOMEPATH"/.ssh/id_ed25519 "$HOMEPATH"/.ssh/known_hosts "$HOME"/.ssh
printf "Print content of .ssh dir\n"
ls -la "$HOME"/.ssh
mkdir program
cd program || exit 2
git clone git@github.com:bakajstep/KNN_Project2024.git
if [ $? != 0 ]; then
  printf "Cloning repository failed!\n"
  exit 1
fi
cd KNN_Project2024 || exit 2
# git checkout "$branch"
cd ../..

# Prepare directory with results
# printf "Prepare directory with results\n"
# if [ ! -d "$HOMEPATH"/program/results/ ]; then # test if dir exists
#   mkdir "$HOMEPATH"/program/ "$HOMEPATH"/program/results/
# fi

# Prepare local directory with results
# mkdir program/results

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/KNN_Project2024 || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

# For now, just run the training script:
printf "Start training\n"
python3 cnec2_ner_trainer.py
printf "Training exit code: %s\n" "$?"

# clean the SCRATCH directory
clean_scratch
