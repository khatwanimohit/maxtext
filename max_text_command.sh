export RUN=0000
export EMBED=$1
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export TPU_PREMAPPED_BUFFER_SIZE=4294967296 
#export XLA_FLAGS="--xla_dump_to=/tmp/hlo_${RUN}_${EMBED}"
#export LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_${RUN}_${EMBED}"
export TPU_VMODULE=3

#python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN base_output_directory=gs://rwitten-x1 vocab_relative_path=vocabs_2 enable_checkpointing=false ici_fsdp_parallelism=16 ici_tensor_parallelism=16 per_device_batch_size=1 steps=10 enable_profiler=true scale=4 dataset_path=gs://tensorflow-datasets/datasets
python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN base_output_directory=gs://rwitten-x1 vocab_relative_path=vocabs_2 enable_checkpointing=false ici_fsdp_parallelism=16 ici_tensor_parallelism=16 per_device_batch_size=1 steps=10 enable_profiler=true scale=4 dataset_path=gs://tensorflow-datasets/datasets