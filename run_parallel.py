import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import datetime

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()
    return process.returncode

def run_commands_parallel(commands, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_command, commands)
    return list(results)



current_time = datetime.datetime.now()
datetime_str = current_time.strftime("%Y%m%d%H%M")
# EXPT_NAME = "dev"
# DB_NAME = "dln_lambdahat_dev"

# EXPT_NAME = f"zero_batch500_width10-50_layer2-15_withtraining_{datetime_str}"
EXPT_NAME = f"randsv_batch500_width15_layer2-5_notraining_funcrank_{datetime_str}"
DB_NAME = "dln_lambdahat"

SGLD_EPSILON = 5e-6
SGLD_NUMSTEPS = 10000
SGLD_BATCH_SIZE = 500
PROP_RANK_REDUCE = 0.8
COMMANDS = []
TRUE_PARAM_METHOD = "rand_rank"  # 'zero' / 'rand_rank' / rand_rank_sv / random
DO_TRAINING = False
DO_FUNCTIONAL_RANK = True
NUM_SEEDS = 100

num_layer = 15
layer_widths = list(np.random.randint(2, 20, size=num_layer))
input_dim = np.random.randint(2, 20)

for seed_i in range(NUM_SEEDS):
    # num_layer = np.random.randint(2, 5)
    # num_layer = 5
    # for width in range(11, 20):
    #     layer_widths = [width] * num_layer
    #     input_dim = width
    
    cmd = [
        f"python expt_dln.py -m localhost:27017:{DB_NAME} with",
        f"expt_name='{EXPT_NAME}'",
        f"layer_widths='{layer_widths}'",
        f"input_dim={input_dim}",
        f"true_param_config.method='{TRUE_PARAM_METHOD}'",
        f"true_param_config.prop_rank_reduce={PROP_RANK_REDUCE}",
        f"sgld_config.epsilon={SGLD_EPSILON}",
        f"sgld_config.num_steps={SGLD_NUMSTEPS}",
        f"sgld_config.batch_size={SGLD_BATCH_SIZE}",
        f"do_training={DO_TRAINING}",
        f"do_functional_rank={DO_FUNCTIONAL_RANK}",
        f"seed={seed_i}"
    ]
    COMMANDS.append(" ".join(cmd))

MAX_WORKERS = 6
print(EXPT_NAME)
print(f"Num experiments: {len(COMMANDS)}")
results = run_commands_parallel(COMMANDS, MAX_WORKERS)
code, freq = np.unique(results, return_counts=True)
print(f"Return codes: {list(code)}, freq: {list(freq)}")