export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3


torchrun \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d  \
--rdzv_endpoint=10.16.78.113:12345 \
tools/train.py --ddp