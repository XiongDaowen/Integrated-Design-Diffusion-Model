export NCCL_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=eno1

torchrun \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d  \
--rdzv_endpoint=10.16.68.163:12345 \
tools/train.py --ddp