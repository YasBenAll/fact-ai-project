python -m experiments.classification.adult_demographic_shift iclr_adult_fixed_ds_rl_di    --n_jobs 24 --n_trials 1 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DisparateImpact    --e  -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --cs_scale 1.5 --fixed_dist --robust_loss
python -m experiments.classification.adult_demographic_shift iclr_adult_antag_ds_rl_di    --n_jobs 24 --n_trials 1 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss