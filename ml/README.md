# Reproduce ML parts

## White box
### Fitting Scan Pipeline (3D Plots)
Train for simple simple scan operator for tables with at least 160mb.
```
python3 white_box_single_pipeline.py
    --pipeline scan
    --min_table_size 160000
    --training_data_path data/simple_sel_pipeline.csv
    --model_path_prefix models/white_box/scan_larger_160k
    --num_epochs 2000
    --batch_size 100
    --param_scale 0.1
```

Train for simple simple scan operator for tables between 40mb and 160mb
```
python3 white_box_single_pipeline.py
    --pipeline scan
    --max_table_size 80000
    --min_table_size 40000
    --training_data_path data/simple_sel_pipeline.csv
    --model_path_prefix models/white_box/scan_smaller_40k_80k
    --num_epochs 10000
    --batch_size 100
    --param_scale 0.1
```

### White Box for Scan - Data Efficiency

Train scan white box model with certain shares of the training data. Note that the expansion factor is not taken into 
account at this point. This has to be done at plotting time. From 2%-5% we see a large improvement.
```
python3 white_box_single_pipeline.py
    --pipeline scan
    --train_sizes 1 2 5 10 20 40 60 80 100
    --min_table_size 160000
    --training_data_path data/simple_sel_pipeline.csv
    --model_path_prefix models/white_box/training_data_share/scan_larger_160k
    --num_epochs 10000
    --batch_size 100
    --param_scale 0.1
    --train_test_split 0.9
```

Evaluate the models, generate Q-errors on the test set and plot the data.
```
python3 white_box_data_efficiency.py
    --plot_path_prefix ./plots/white_box/training_data_share
    --csv_path data/simple_sel_pipeline.csv
    --target_path ./plots/white_box/white_box_data_efficiency.csv
    --train_sizes 1 2 5 10 20 40 60 80 100
    --min_table_size 160000
    --train_test_split 0.9
    --model simple_scan
```

You can then use the data_efficiency.ipynb notebook to generate the according plots.

### White Box Generalization
```
python3 white_box_single_pipeline.py
    --pipeline scan
    --min_table_size 160000
    --max_table_size 700000
    --excluded_table_sizes 320000
    --training_data_path data/simple_sel_pipeline.csv
    --model_path_prefix models/white_box/scan_larger_160k_generalization
    --num_epochs 2000
    --batch_size 100
    --param_scale 0.1
```

### White Box for Join Query - Data Efficiency

Learn a white box model for the build pipeline
```
python3 white_box_single_pipeline.py
    --pipeline build
    --training_data_path data/build_pipeline.csv
    --model_path_prefix models/white_box/build
    --train_sizes 1 2 5 10 20 40 60 80 100
    --num_epochs 2000
    --batch_size 100
    --param_scale 0.1
    --train_test_split 0.9
```

Learn a white box model for the probe pipeline
```
python3 white_box_single_pipeline.py
    --pipeline probe
    --training_data_path data/probe_pipeline.csv
    --model_path_prefix models/white_box/probe
    --train_sizes 1 2 5 10 20 40 60 80 100
    --num_epochs 2000
    --batch_size 100
    --param_scale 0.1
    --train_test_split 0.9
```

Evaluate the models, generate Q-errors on the test set and plot the data.
```
python3 white_box_data_efficiency.py
    --plot_path_prefix ./plots/white_box/join_training_data_share
    --csv_path data/join_query.csv
    --target_path ./plots/white_box/white_box_join_data_efficiency.csv
    --train_sizes 1 2 5 10 20 40 60 80 100
    --min_table_size 160000
    --train_test_split 0.9
    --model join_query
```


Evaluate the models on probe pipeline only, generate Q-errors on the test set and plot the data.
```
python3 white_box_data_efficiency.py
    --plot_path_prefix ./plots/white_box/join_training_data_share
    --csv_path data/probe_pipeline.csv
    --target_path ./plots/white_box/white_box_probe_data_efficiency.csv
    --train_sizes 1 2 5 10 20 40 60 80 100
    --min_table_size 160000
    --train_test_split 0.9
    --model probe
```


Evaluate the models on build pipeline only, generate Q-errors on the test set and plot the data.
```
python3 white_box_data_efficiency.py
    --plot_path_prefix ./plots/white_box/join_training_data_share
    --csv_path data/build_pipeline.csv
    --target_path ./plots/white_box/white_box_build_data_efficiency.csv
    --train_sizes 1 2 5 10 20 40 60 80 100
    --min_table_size 160000
    --train_test_split 0.9
    --model build
```


## Black Box Simple Scan Data Efficiency

Train scan black box model with certain shares of the training data.
```
python3 black_box.py
    --base_path ./
    --experiment_name simple_sel_pipeline
    --train_sizes 2 4 6 8 10 20 40 80 100
    --num_epochs 4000
    --batch_size 100
    --param_scale 0.1
    --step_size 0.01
    --table_size_threshold 160000
```

Evaluate the q-errors.
```
python3 black_box_data_efficiency.py
    --no_operators 3
    --no_sample_tuples 4
    --dim_predicate_embedding 3
    --base_path ./
    --experiment_name simple_sel_pipeline
    --train_sizes 2 4 6 8 10 20 40 80 100
    --min_table_size 160000
```

You can then use the data_efficiency.ipynb notebook to generate the according plots.

## Black Box Join Query Data Efficiency

Train scan black box model with certain shares of the training data.
```
python3 black_box.py
    --base_path ./
    --experiment_name query_plan_stats
    --train_sizes 2 4 6 8 10 20 40 80 100
    --num_epochs 4000
    --batch_size 100
    --param_scale 0.1
    --step_size 0.01
    --table_size_threshold 160000
```

Evaluate the q-errors.
```
python3 black_box_data_efficiency.py
    --no_operators 3
    --no_sample_tuples 4
    --dim_predicate_embedding 3
    --base_path ./
    --experiment_name query_plan_stats
    --train_sizes 2 4 6 8 10 20 40 80 100
    --min_table_size 160000
```

## Adversarial

Done by Aditya.
https://github.com/DataManagementLab/adversial_learning

