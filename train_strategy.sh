bash stop_trainer.sh

factor=0.95
lr=0.001
epoch_step=5
for ((epoch=0;epoch<=120;epoch+=$epoch_step)); do
    cur_lr=$(python -c "print($lr*($factor)**$epoch)")
    end_epoch=$(python -c "print($epoch + $epoch_step)")
    python train.py --lr $cur_lr --start_from_epoch $epoch --epochs $end_epoch --preload_data true
    bash stop_trainer.sh
done