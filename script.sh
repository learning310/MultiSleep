# ------------------------------------------------------------------------------------------------------------------------
# for iter in 10 20 30 40 50 60 
# do
# python single.py --logs_save_dir experiments_20 --experiment_description Exp1_epoch --run_description epoch_$iter --dataset edf20 --epochs $iter
# done

# for iter in 10 20 30 40 50 60
# do
# python single.py --logs_save_dir experiments_78 --experiment_description Exp1_epoch --run_description epoch_$iter --dataset edf78 --epochs $iter
# done
# ------------------------------------------------------------------------------------------------------------------------

python multi.py --logs_save_dir experiments_20 --experiment_description Exp3 --dataset edf20
python multi.py --logs_save_dir experiments_78 --experiment_description Exp3 --dataset edf78