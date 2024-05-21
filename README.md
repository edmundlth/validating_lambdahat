# Validating local learning coefficient estimation algorithm
This repository contain code to run a few types of experiments verifying that an algorithm that estimates a geometric quantity known as the *learning coefficient* for machine learning models is scalably accurate and can detect geometrical changes over model training. 

The experimental findings are reported in [ANONYMISED]

# Installation 
Installation: Use [pipenv](https://pipenv.pypa.io/en/latest/) to install required packages by running `pipenv install` in the top directory containing `Pipfile.lock` and start up the virtual environment with `pipenv shell`. 

Alternatively, run `pip install -r requirements.txt` in the top directory. 

Note that experiment configuration and recording is handled by the [Sacred](https://sacred.readthedocs.io/en/stable/index.html) library. See [their docs](https://sacred.readthedocs.io/en/stable/observers.html) for configuring the database or file system used for experiment logging. 


# Running Experiments
## Experiment: $\hat{\lambda}(w^*)$-vs-$\lambda$ experiments for Deep Linear Networks (DLN)
A sample command to run a single experiment is 
```
python expt_dln.py -F </PATH/TO/OUTPUT/DIRECTORY/> with expt_name='<CURRENT_EXPERIMENT_NAME>' num_training_data=<1000000> layer_widths='<[259, 286, 885, 235, 204, 274, 252]>' input_dim=<185> true_param_config.method='<rand_rank>' true_param_config.prop_rank_reduce=<0.5> sgld_config.epsilon=<2e-07> sgld_config.num_steps=<20000> sgld_config.gamma=<1.0> sgld_config.batch_size=<500> do_training=<True> training_config.optim='<sgd>' training_config.learning_rate=<0.01> training_config.batch_size=<500> training_config.num_steps=<50000> do_functional_rank=<False> do_hessian_trace=<False> seed=<1>
```
Values in angle brackets, `<VALUE>` are example experiment configuration. Replace `-F </PATH/TO/OUTPUT/DIRECTORY/>` with `-m mongo_db_url:port:db_name` to use Sacred's MongoDB observer instead. E.g. `-m localhost:27017:dln_saddle_dynamics` if you have MongoDB running locally. See full set of configuration in `expt_dln.py`. 


To run many experiments with a few configuration randomly sample from a specified range, modify the global configuration variables (with all captial letters variable names) in `gen_commands_expt_dln.py` and run 
```
python gen_commands_expt_dln.py </PATH/TO/OUTPUTDIRECTORY/COMMANDS.txt>
```
To generate a list of commands to be run on a cluster with SLURM or other compute platform.

See `job.slurm` for the [SLURM](https://slurm.schedmd.com/sbatch.html) job script to submit the generated list of commands to run on a SLURM-managed HPC cluster with GPU. Modify the path to `<COMMANDS.txt>` file in `job.slurm` to the command list generated above. 


## ResNet18 + CIFAR10 $\hat{\lambda}(w^*)$ over SGD training
A sample command to run a single experiment is 
```
python expt_llc_curve.py --comment <ANY COMMENTS> -F </PATH/TO/OUTPUT/DIRECTORY/> with expt_name='<CURRENT EXPERIMENT NAME>' sgld_config.epsilon=2e-07 sgld_config.gamma=1.0 sgld_config.num_steps=5000 sgld_config.batch_size=2048 loss_trace_minibatch=True training_config.optim=sgd training_config.momentum=None training_config.num_steps=30001 training_config.learning_rate=0.005 training_config.batch_size=512 training_config.l2_regularization=None force_realisable=False logging_period=1000 do_plot=False verbose=False seed=0
```
Values in angle brackets, `<VALUE>` are example experiment configuration. Replace `-F </PATH/TO/OUTPUT/DIRECTORY/>` with `-m mongo_db_url:port:db_name` to use Sacred's MongoDB observer instead. E.g. `-m localhost:27017:expt_resnet` if you have MongoDB running locally. See full set of configuration in `expt_llc_curve.py`. 

To generate a list of commands with a combination of configuration, modify the `config` dictionary in `gen_commands_expt_llc_curve.py` and run 
```
python gen_commands_expt_llc_curve.py </PATH/TO/OUTPUTDIRECTORY/COMMANDS.txt>
```



## Tracking $\hat{\lambda}(w^*)$ over SGD training for Deep Linear Networks (DLN)
This reproduces the experimental setup reported in 
```
Jacot, Arthur, François Ged, Berfin Şimşek, Clément Hongler, and Franck Gabriel. 2021. “Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity.” arXiv [stat.ML]. arXiv. http://arxiv.org/abs/2106.15933.
```
with the addition that we track the local learning coefficient, $\hat{\lambda}(w^*)$, over the course of training, expecting it to detect local geometrical changes as the training trajectory move from 'saddle-to-saddle'. 

Commands for running one experiment: 
```
python expt_dln_saddle_dynamics.py -m <MONGODBURL:27017:DB_NAME> with expt_name=<EXPT_NAME> width=<500> seed=<42> training_config.num_steps=<50000> logging_period=<100> 
```
See full set of config options in `expt_dln_saddle_dynamics.py`. 

## $\hat{\lambda}(w^*)$ Rescaling-invariance
This experiment is contained within the `rescaling_invariance.ipynb` Jupyter notebook. 

## Benchmarking SGLD agaist full-gradient MCMC method
This experiment is contained within the `MALA_vs_SGLD.ipynb` Jupyter notebook. 



# Generate Plots
Once experiments are finished, the plots as seen in the paper is generated by running `plots_final.ipynb` from top to bottom with 
- the variable `directory_paths` containing the output directories specified in `COMMANDS.txt` above
- the variables `db` and `client` pointing to the right MongoDB urls if they are used. 
- the variable `expts` containing `EXPT_NAME`s used in the `expt_dln_saddle_dynamics.py` experimental runs. 
- the variable `EXPERIMENT_RECORDS` should contain a list of dictionaries specifying the `EXPT_NAME`, what variable to vary with `hue_var` and how to filter the data with `filters`.
