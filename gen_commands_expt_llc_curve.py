"""
Usage: python gen_commands_expt_llc_curve.py <output_filepath>
"""

import datetime
import sys
import itertools
import os
import glob

def unroll_config(config):
    # Prepare lists for keys and values, where values are always lists (even if singleton)
    keys = []
    value_lists = []
    
    for key, value in config.items():
        keys.append(key)
        if isinstance(value, list):
            value_lists.append(value)
        else:
            value_lists.append([value])  # Make it a list to handle uniformly in product

    # Use itertools.product to generate all combinations of parameter values
    unrolled_configs = []
    for values in itertools.product(*value_lists):
        # Create a dictionary for each combination
        new_config = dict(zip(keys, values))
        unrolled_configs.append(new_config)

    return unrolled_configs

current_time = datetime.datetime.now()
datetime_str = current_time.strftime("%Y%m%d%H%M")

EXPT_NAME = f"varylabelnoise_{datetime_str}"
# EXPT_NAME = f"expt_llc_curve_batch{SGLD_BATCH_SIZE}_eps{SGLD_EPSILON}_nstep{SGLD_NUMSTEPS}_{datetime_str}"

# DB_NAME = "expt_llc_curve"
# SACRED_OBSERVER = "-m localhost:27017:{DB_NAME}"
SACRED_OBSERVER = f"-F ./outputs/expt_llc_curve_outputs/{EXPT_NAME}/"


config = {
    "expt_name": EXPT_NAME,
    "sgld_config.epsilon": [1e-7],
    "sgld_config.gamma": 1.0,
    "sgld_config.num_steps": 2000,
    "sgld_config.batch_size": 2048,
    "loss_trace_minibatch": True,

    "training_config.optim": "sgd", # ["sgd", "adam"], 
    "training_config.momentum": [None, 0.9], # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "training_config.num_steps": 100001,
    "training_config.learning_rate": 0.005, # [0.005, 0.01], # [0.005, 0.01, 0.05, 0.1, 0.2], # 0.2
    "training_config.batch_size": [16, 512], # [8, 16, 32, 64, 128, 256, 512, 1024], # [64, 128, 256, 512, 1024], # 512, # [64, 512], 
    "training_config.l2_regularization": None, # [0.0, 0.01, 0.025, 0.05, 0.075, 0.1], # None
    "model_data_config.label_noise_level": [0, 0.05, 0.1, 0.15, 0.25],

    "force_realisable": False, 
    "logging_period": 2500,
    "do_plot": False,
    "verbose": False,
    "seed": [0, 1, 2, 3, 4],
}

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("Please Specify output filepath.")
    
    filepath = sys.argv[1]
    overwrite = sys.argv[2] if len(sys.argv) > 2 else "n"
    if os.path.exists(filepath) and not overwrite.lower().startswith("y"):
        # files with the same filename substring excluding extension
        basename = os.path.basename(filepath).split(".")[0]
        files_with_same_filenames = '\n'.join(
            sorted(glob.glob(f"{basename}*.txt", root_dir=os.path.dirname(filepath)))
        )
        raise RuntimeError(
            f"File already exists at {filepath}.\n"
            "Please specify a different filepath or give the `y` flag.\n"
            f"Existing files:\n{files_with_same_filenames}"
        )

    unrolled_config = unroll_config(config)
    COMMANDS = []
    for config_i in unrolled_config:
        if config_i["training_config.optim"] == "adam" and config_i["training_config.momentum"] is not None: # Skip Adam with momentum
            continue
        cmd = [
            f"python expt_llc_curve.py --comment {filepath} {SACRED_OBSERVER} with",
            *[f"{key}={value}" for key, value in config_i.items()]
        ]
        COMMANDS.append(" ".join(cmd))
    with open(sys.argv[1], "w") as outfile:
        outfile.write('\n'.join(COMMANDS))
    print(f"Generated {len(COMMANDS)} commands.")
    