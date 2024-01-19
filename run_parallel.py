import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import datetime
import sys

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
# DB_NAME = "dln_lambdahat_dev"
DB_NAME = "dln_lambdahat"


SGLD_EPSILON = 1e-7
SGLD_NUMSTEPS = 50000
SGLD_BATCH_SIZE = 500
SGLD_GAMMA = 1.0
PROP_RANK_REDUCE = 0.5
COMMANDS = []
TRUE_PARAM_METHOD = "rand_rank"  # 'zero' / 'rand_rank' / rand_rank_sv / random
DO_TRAINING = True
DO_FUNCTIONAL_RANK = False
DO_HESSIAN_TRACE = False
NUM_SEEDS = 100
WIDTHMIN, WIDTHMAX = 500, 1000
NLAYERMIN, NLAYERMAX = 2, 20


# EXPT_NAME = "dev_large"
# EXPT_NAME = f"zero_batch500_width10-50_layer2-15_withtraining_{datetime_str}"
# EXPT_NAME = f"randsv_batch500_width15_layer2-5_notraining_funcrank_{datetime_str}"
# EXPT_NAME = f"randrank_batch500_width10-30_layer5_notraining_funcrank_hesstrace_{datetime_str}"
EXPT_NAME = f"{TRUE_PARAM_METHOD}_batch{SGLD_BATCH_SIZE}_width{WIDTHMIN}-{WIDTHMAX}_layer{NLAYERMIN}-{NLAYERMAX}_train{DO_TRAINING}_{datetime_str}"


# SACRED_OBSERVER = "-m localhost:27017:{DB_NAME}"
SACRED_OBSERVER = f"-F ./outputs/{EXPT_NAME}/"


for seed_i in range(NUM_SEEDS):
    num_layer = np.random.randint(NLAYERMIN, NLAYERMAX)
    layer_widths = list(np.random.randint(WIDTHMIN, WIDTHMAX, size=num_layer))
    input_dim = np.random.randint(WIDTHMIN, WIDTHMAX)
    
    cmd = [
        f"python expt_dln.py {SACRED_OBSERVER} with",
        f"expt_name='{EXPT_NAME}'",
        f"layer_widths='{layer_widths}'",
        f"input_dim={input_dim}",
        f"true_param_config.method='{TRUE_PARAM_METHOD}'",
        f"true_param_config.prop_rank_reduce={PROP_RANK_REDUCE}",
        f"sgld_config.epsilon={SGLD_EPSILON}",
        f"sgld_config.num_steps={SGLD_NUMSTEPS}",
        f"sgld_config.gamma={SGLD_GAMMA}",
        f"sgld_config.batch_size={SGLD_BATCH_SIZE}",
        f"do_training={DO_TRAINING}",
        f"training_config.optim='adam'",
        f"training_config.learning_rate=0.01",
        f"training_config.batch_size=500",
        f"training_config.num_steps=10000",
        f"do_functional_rank={DO_FUNCTIONAL_RANK}",
        f"do_hessian_trace={DO_HESSIAN_TRACE}",
        f"seed={seed_i}"
    ]
    COMMANDS.append(" ".join(cmd))

if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as outfile:
        outfile.write('\n'.join(COMMANDS))
    sys.exit()

MAX_WORKERS = 6
print(EXPT_NAME)
print(f"Num experiments: {len(COMMANDS)}")
results = run_commands_parallel(COMMANDS, MAX_WORKERS)
code, freq = np.unique(results, return_counts=True)
print(f"Return codes: {list(code)}, freq: {list(freq)}")