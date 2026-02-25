#!/bin/bash
#SBATCH --job-name=ChemDFM_streamer         # Job name
#SBATCH --output=logs/ChemDFM_%A_%a.out     # Standard output file (jobID_arrayID)
#SBATCH --error=logs/ChemDFM_%A_%a.err      # Standard error file (jobID_arrayID)
#SBATCH --array=1-1%1                       # Array job range
#SBATCH --mem=32G                           # Memory allocation
#SBATCH --time=00:10:00                     # Maximum run time (hh:mm:ss)
#SBATCH --cpus-per-task=1                   # CPUs per task
#SBATCH --nodelist=epyc-A40     
#SBATCH --partition=gpu.q                     

######## Only edit the paths in the block below. ########

working_dir=$(pwd)
script_batch='llm_batch_execute.py'
script_streamer='llm_batch_execute_streamer.py'
promptfile='prompts/analyzeAbstract.txt'
abstractsfile='brown_abstracts/brown_allExamples_w_abstracts.csv'

######## No need to edit anything below this line. ########

start_time=$(date "+%s")
echo "Executing ChemDFM streamer script"
echo "Start: $start_time s"
echo "This job is running on node(s): $SLURM_NODELIST"
source /nfs/home/zdingman/environments/LLM_analysis/bin/activate || exit 1
cd "$working_dir" || exit 1
echo "Working dir: $working_dir"

echo "This job is querying $abstractsfile"

batch_start_time=$(date "+%s")
python $script_batch $promptfile $abstractsfile
streamer_start_time=$(date "+%s")
python $script_streamer $promptfile $abstractsfile

end_time=$(date "+%s")
echo "End: $end_time s"
runtime=$((end_time - start_time))
echo "Runtime: $runtime s"
batch_runtime=$((streamer_start_time - batch_start_time))
echo "Batch runtime: $batch_runtime s"
streamer_runtime=$((end_time - streamer_start_time))
echo "Streamer runtime: $streamer_runtime s"
