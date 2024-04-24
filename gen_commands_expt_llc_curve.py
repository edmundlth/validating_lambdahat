"""
Usage: python gen_commands_expt_llc_curve.py <output_filepath>
"""

import datetime
import sys
import itertools

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

EXPT_NAME = "dev"
# EXPT_NAME = f"expt_llc_curve_batch{SGLD_BATCH_SIZE}_eps{SGLD_EPSILON}_nstep{SGLD_NUMSTEPS}_{datetime_str}"

# DB_NAME = "expt_llc_curve"
# SACRED_OBSERVER = "-m localhost:27017:{DB_NAME}"
SACRED_OBSERVER = f"-F ./outputs/expt_llc_curve_outputs/{EXPT_NAME}/"


config = {
    "expt_name": EXPT_NAME,
    "sgld_config.epsilon": 5e-6,
    "sgld_config.gamma": 1.0,
    "sgld_config.num_steps": 5001,
    "sgld_config.batch_size": 2048,
    "training_config.optim": ["sgd", "adam"],
    "training_config.learning_rate": [1e-3, 1e-4],
    "training_config.momentum": [None, 0.9],
    "training_config.batch_size": [128, 512],
    "training_config.num_steps": 10001,
    "force_realisable": False,
    "logging_period": 500,
    "seed": [0, 1]
}

unrolled_config = unroll_config(config)
COMMANDS = []
for config_i in unrolled_config:
    cmd = [
        f"python expt_llc_curve.py {SACRED_OBSERVER} with",
        *[f"{key}={value}" for key, value in config_i.items()]
    ]
    COMMANDS.append(" ".join(cmd))

print(f"Generated {len(COMMANDS)} commands.")
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as outfile:
        outfile.write('\n'.join(COMMANDS))
else:
    raise RuntimeError("Please Specify output filepath.")