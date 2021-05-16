bash stop_trainer.sh

train_images_path=/home/evjeny/data_dir/perimetry_text_detection_split/train_images
train_gts_path=/home/evjeny/data_dir/perimetry_text_detection_split/train_gts

test_images_path=/home/evjeny/data_dir/doctors_research_base/images
save_images_path=/home/evjeny/workspace/tonometry_notebooks/perimetry_text_detection/east

factor=0.95
lr=0.001
epoch_step=5
for ((epoch=0;epoch<=120;epoch+=$epoch_step)); do
    cur_lr=$(python -c "print($lr*($factor)**$epoch)")
    end_epoch=$(python -c "print($epoch + $epoch_step)")

    python train.py --lr $cur_lr --start_from_epoch $epoch --epochs $end_epoch --preload_data \
        --images_path $train_images_path --gts_path $train_gts_path
    python detect.py --images_path $test_images_path \
        --save_path "$save_images_path/epoch_$end_epoch" \
        --weights "pths/model_epoch_$end_epoch.pth" --subset_size 30 --device cuda
    bash stop_trainer.sh
done