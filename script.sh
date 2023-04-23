# ------------------------------------------------------------------------------------------------------------------------
for iter in 10 20 30 40 50
do
python single.py --logs_save_dir experiments_78 --experiment_description TimeTransformer_epoch --run_description epoch_$iter --dataset edf78 --epochs $iter
done

# for ratio in 1 3 5 7
# do
# python single.py --experiment_description TimeTransformer_dropout --run_description ratio_$ratio --dataset edf20 --epochs $ratio
# done

# ------------------------------------------------------------------------------------------------------------------------
# python single.py --run_description SingleEEG --dataset edf20 --modality eeg
# python single.py --run_description SingleEEG --dataset edf78 --modality eeg
# python single.py --run_description SingleEOG --dataset edf20 --modality eog
# python single.py --run_description SingleEOG --dataset edf78 --modality eog

# for iter in 10 20 30 40 50
# do
# python multi.py --logs_save_dir experiments_20 --experiment_description MultiSleep_epoch --run_description epoch_$iter --dataset edf20 --epochs $iter
# done

# python multi.py -run_description --dataset edf20