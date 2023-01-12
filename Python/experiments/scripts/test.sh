python -m experiments.classification.brazil_demographic_shift iclr_brazil_fixed_ds_rl_di    --n_jobs 1 --n_trials 25 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --gpa_cutoff 3.0 --dshift_var race --cs_scale 1.5 --fixed_dist --robust_loss
