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
current_date = current_time.strftime("%Y%m%d")
# EXPT_NAME = "dev"
# DB_NAME = "dln_lambdahat_dev"

EXPT_NAME = f"random_sv_{current_date}"
DB_NAME = "dln_lambdahat"

SGLD_NUMSTEPS = 20000
COMMANDS = []
NUM_EXPERIMENTS = 100
for expt_i in range(NUM_EXPERIMENTS):
    num_layer = np.random.randint(2, 20)
    layer_widths = list(np.random.randint(5, 30, size=num_layer))
    input_dim = np.random.randint(5, 20)
    cmd = [
        f"python expt_dln.py -m localhost:27017:{DB_NAME} with",
        f"expt_name='{EXPT_NAME}'",
        f"layer_widths='{layer_widths}'",
        f"input_dim={input_dim}",
        f"true_param_config.method='rand_rank_sv'",
        f"sgld_config.num_steps={SGLD_NUMSTEPS}",
        f"seed={expt_i}"
    ]
    COMMANDS.append(" ".join(cmd))

MAX_WORKERS = 5
results = run_commands_parallel(COMMANDS, MAX_WORKERS)
print(np.unique(results, return_counts=True))
