echo "Running 32b.sh"
# 32B parameter model.
# This config will work out of the box for any number of v5p-128 slices.
#
# Command Flags:
# OUTPUT_PATH (Required, unless base_output_directory is already set in base.yml)
# DATASET_PATH (Required, unless dataset_path is already set in base.yml)
# RUN_NAME (Required, unless run_name is already set in base.yml or running with XPK/GKE)
# PLATFORM (Optional, can be "gke" or "gce", default is "gce")
#
# Example to invoke this script:
# bash MaxText/configs/v5p/32b.sh RUN_NAME="<your_run_name>" OUTPUT_PATH="gs://<your_output_path>" DATASET_PATH="gs://<your_dataset_path>" PLATFORM="gke"


# Stop execution if any command exits with error
set -e

export PLATFORM="gce"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

# Set up network optimizations
bash preflight.sh PLATFORM=$PLATFORM

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=30 per_device_batch_size=8 enable_checkpointing=false\
    enable_profiler=false remat_policy=minimal global_parameter_scale=32\
    ici_fsdp_parallelism=-1 ici_tensor_parallelism=4\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH use_iota_embed=true reuse_example_batch=1\
    dataset_type=synthetic gcs_metrics=true attention='flash' int8_training=false\
