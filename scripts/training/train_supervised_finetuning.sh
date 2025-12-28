#!/bin/bash
# Navigate to project root directory
cd "$(dirname "$0")/../.." || exit 1

# Supervised Finetuning Script - Multiple Properties with Checkpoint vs Random Init
# Compares pretrained CLaC model with random initialization

# Pretrained checkpoint path
CHECKPOINT_PATH="outputs/your-model.ckpt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Properties to test
PROPERTIES=(
    "mbj_bandgap"
    "slme"
    "magmom_outcar"
    "p-Seebeck"
)

# Training configurations
MAX_EPOCHS=300
BATCH_SIZE=64
DEBUG=false

echo "=========================================="
echo "Supervised Finetuning Benchmark"
echo "Dataset: dft_3d_gpt_narratives"
echo "Max Epochs: $MAX_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Loop through each property
for PROPERTY in "${PROPERTIES[@]}"; do
    # Clean property name for file/directory naming
    CLEAN_NAME=$(echo "$PROPERTY" | sed 's/ /_/g' | sed 's/(//g' | sed 's/)//g' | sed 's/\///g')

    echo "================================================"
    echo "Testing Property: $PROPERTY"
    echo "================================================"

    # 1. Train with pretrained checkpoint
    echo ""
    echo ">>> [1/2] Training with PRETRAINED checkpoint <<<"
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo ""

    python train.py \
        model=supervised_finetuning \
        model.fine_tune_from="$CHECKPOINT_PATH" \
        model.base_model_type=clac \
        model.freeze_graph_encoder=false \
        model.datamodule.label="$PROPERTY" \
        hyperparams.batch_size=$BATCH_SIZE \
        model.optimizer.optimizer.lr=0.0001 \
        model.optimizer.use_lr_scheduler=true \
        trainer.pl_trainer.max_epochs=$MAX_EPOCHS \
        trainer.pl_trainer.accumulate_grad_batches=1 \
        trainer.pl_trainer.devices=1 \
        trainer.pl_trainer.gradient_clip_val=0.005 \
        exp_name=supervised-${CLEAN_NAME}-painn-pretrained \
        hydra.run.dir=outputs/supervised-${CLEAN_NAME}-painn-pretrained \
        debug=$DEBUG

    if [ $? -ne 0 ]; then
        echo "Error: Training with checkpoint failed for property: $PROPERTY"
        continue
    fi

    echo ""
    echo "✓ Checkpoint training completed for: $PROPERTY"
    echo ""

    # 2. Train with random initialization
    echo ">>> [2/2] Training with RANDOM initialization <<<"
    echo "Checkpoint: null (random init)"
    echo ""

    python train.py \
        model=supervised_finetuning \
        model.fine_tune_from=null \
        model.base_model_type=clac \
        model.freeze_graph_encoder=false \
        model.datamodule.label="$PROPERTY" \
        hyperparams.batch_size=$BATCH_SIZE \
        model.optimizer.optimizer.lr=0.0001 \
        model.optimizer.use_lr_scheduler=true \
        trainer.pl_trainer.max_epochs=$MAX_EPOCHS \
        trainer.pl_trainer.accumulate_grad_batches=1 \
        trainer.pl_trainer.gradient_clip_val=0.005 \
        trainer.pl_trainer.devices=1 \
        exp_name=supervised-${CLEAN_NAME}-painn-random \
        hydra.run.dir=outputs/supervised-${CLEAN_NAME}-painn-random \
        debug=$DEBUG

    if [ $? -ne 0 ]; then
        echo "Error: Training with random init failed for property: $PROPERTY"
        continue
    fi

    echo ""
    echo "✓ Random init training completed for: $PROPERTY"
    echo ""
    echo "================================================"
    echo ""
done

echo "=========================================="
echo "All Training Completed!"
echo "=========================================="
echo ""
echo "Results saved in outputs/supervised-* directories"
