echo "Running 64b.sh"
# Example command to invoke this script
# bash MaxText/configs/experimental/64b.sh

# Stop execution if any command exits with error
set -e

export OUTPUT_PATH="gs://maxtext-experiments-multipod"
export DATASET_PATH="gs://maxtext-dataset/"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

if [[ $PLATFORM == "gce" ]]; then
    # Set up network for running on gce
    bash rto_setup.sh
    # For DNS lookup when running on large number of VMs on gce
    echo '142.250.123.95 www.googleapis.com' | sudo tee -a /etc/hosts
    echo '142.251.4.128 storage.googleapis.com' | sudo tee -a /etc/hosts
else
    # Set up network for running on gke
    bash gke_rto_setup.sh
    # For DNS lookup when running on large number of VMs on gke
    echo '142.250.123.95 www.googleapis.com' | tee -a /etc/hosts
    echo '142.251.4.128 storage.googleapis.com' | tee -a /etc/hosts
fi

# Train
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME\
    steps=30 per_device_batch_size=4 enable_checkpointing=false\
    enable_profiler=false remat_policy=minimal global_parameter_scale=64\
    ici_fsdp_parallelism=-1 ici_tensor_parallelism=4\
    max_target_length=2048 base_output_directory=$OUTPUT_PATH\
    dataset_path=$DATASET_PATH embed_lookup_style='iota' reuse_example_batch=1\
    dataset_type=synthetic gcs_metrics=true attention='flash' int8_training=false\