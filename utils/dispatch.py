import argparse
from datetime import datetime
import os 
import re
import shutil
import subprocess
import sys
import time
import warnings


"""
A parser that collects all cluster-specific arguments. To be merged into
an experiment's parser via `parents=[dispatch_parser]`.
"""
def dispatch_argument_parser(*args, **kwargs):
    parser = argparse.ArgumentParser(*args, **kwargs)
    g_sel = parser.add_argument_group('Cluster select.')
    mutex_cluster = g_sel.add_mutually_exclusive_group()
    mutex_cluster.add_argument('--dispatch_canada', action='store_true')
    g_cca = parser.add_argument_group('Compute Canada cluster')
    g_cca.add_argument('--account', type=str, default='rrg-bengioy-ad',
                       choices=['rrg-bengioy-ad', 'def-bengioy'],
                       help="Prefer rrg over def for higher priority.")
    g_cca.add_argument('--cca_gpu', type=int, default=1)
    g_cca.add_argument('--cca_cpu', type=int, default=8)
    g_cca.add_argument('--cca_mem', type=str, default='32G')
    g_cca.add_argument('--time', type=str, default='1-00:00',
                       help="Max run time (DD-HH:MM). Shorter times get "
                            "higher priority.")
    g_cca.add_argument('--copy_local', action='store_true',
                       help="Copy \'data\' to the local scratch space.")
    return parser


"""
Given a parser that contains _all_ of an experiment's arguments (including
the cluster-specific arguments from `dispatch_parser`), as well as a run()
method, run the experiment on the specified cluster or locally if no cluster
is specified.

NOTE: args must contain `path`.
"""
def dispatch(parser, run):
    # Get arguments.
    args = parser.parse_args()
    assert hasattr(args, 'path')
    
    # If resuming, merge with loaded arguments (newly passed arguments
    # override loaded arguments).
    if os.path.exists(os.path.join(args.path, "args.txt")):
        with open(os.path.join(args.path, "args.txt"), 'r') as f:
            saved_args = f.read().split('\n')[1:]
            args = parser.parse_args(args=saved_args)
        args = parser.parse_args(namespace=args)
    
    # Dispatch on a cluster (or run locally if none specified).
    if args.dispatch_canada:
        if _isrunning_canada(args.path):
            # If a job is already running on this path, exit.
            print("WARNING: aborting dispatch since there is already an "
                  "active job ({}).".format(args.path))
            return
        import daemon
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        daemon_log_file = open(os.path.join(args.path, "daemon_log.txt"), 'a')
        print("Dispatch on Compute Canada - daemonizing ({})."
              "".format(args.path))
        with daemon.DaemonContext(stdout=daemon_log_file,
                                  working_directory=args.path):
            _dispatch_canada_daemon(args)
    elif args.model_from is None and not os.path.exists(args.path):
        parser.print_help()
    else:
        if args.copy_local:
            # Copy 'data' to local scratch space.
            assert hasattr(args, 'data')
            target = os.path.join(os.environ["SLURM_TMPDIR"],
                                  os.path.basename(args.data))
            if os.path.exists(target):
                warnings.warn("{} exists - not copying".format(target))
            else:
                if os.path.isdir(args.data):
                    shutil.copytree(args.data, target)
                else:
                    shutil.copyfile(args.data, target)
                args.data = target
        run(args)


def _dispatch_canada(args):
    pre_cmd = ("cd /scratch/veugene/ssl-seg-eugene\n"
               "source register_submodules.sh\n"
               "source activate genseg\n")
    cmd = subprocess.list2cmdline(sys.argv)       # Shell executable.
    cmd = cmd.replace(" --dispatch_canada",   "") # Remove recursion.
    cmd = "#!/bin/bash\n {}\n python3 {}".format(pre_cmd, cmd)  # Combine.
    out = subprocess.check_output([
                    "sbatch",
                    "--account", args.account,
                    "--gres", 'gpu:{}'.format(args.cca_gpu),
                    "--cpus-per-task", str(args.cca_cpu),
                    "--mem", args.cca_mem,
                    "--time", args.time],
                   input=cmd,
                   encoding='utf-8')
    print(out)
    return out


def _dispatch_canada_daemon(args):
    # Parse time argument to seconds.
    if   args.time.count(':')==0 and args.time.count('-')==0:
        # minutes
        time_format = "%M"
    elif args.time.count(':')==0 and args.time.count('-')==1:
        # days-hours
        time_format = "%d-%H"
    elif args.time.count(':')==1 and args.time.count('-')==0:
        # minutes:seconds
        time_format = "%M:%S"
    elif args.time.count(':')==1 and args.time.count('-')==1:
        # days-hours:minutes
        time_format = "%d-%H:%M"
    elif args.time.count(':')==2 and args.time.count('-')==0:
        # hours:minutes:seconds
        time_format = "%H:%M:%S"
    elif args.time.count(':')==2 and args.time.count('-')==1:
        # days-hours:minutes:seconds
        time_format = "%d-%H:%M:%S"
    else:
        raise ValueError("Invalid `time` format ({}).".format(args.time))
    datetime_obj = datetime.strptime(args.time, time_format)
    time_seconds = ( datetime_obj
                    -datetime(datetime_obj.year, 1, 1)).total_seconds()
    if '-' in args.time:
        # Add a day if days are specified, since days count up from 1.
        time_seconds += 24*60*60
    
    # Periodically check status of job. Relaunch on TIMEOUT.
    status = 'TIMEOUT'
    while status=='TIMEOUT':
        # Launch.
        sbatch_output = _dispatch_canada(args)
        job_id = sbatch_output.split(' ')[-1].strip(' \n\t')
        try:
            int(job_id)
        except ValueError:
            raise RuntimeError("Cannot extract job ID from `sbatch` standard "
                               "output: {}".format(sbatch_output))
        
        # Wait until the job is launched before setting a timer.
        status = _get_status_canada(job_id)
        while status=='PENDING' or status is None:
            time.sleep(10)
            status = _get_status_canada(job_id)
        
        # Wait.
        time.sleep(time_seconds)
        
        # Check status.
        status = _get_status_canada(job_id)
        
        # If the job is still RUNNING, wait until the state changes.
        # 
        # The job may continue running while the cluster waits for it
        # to exit after a TERM signal (or until it eventually KILLs the job).
        while status=='RUNNING':
            time.sleep(10)
            status = _get_status_canada(job_id)


def _get_status_canada(job_id):
    status = None
    sacct_output = subprocess.check_output(["sacct", "-j", job_id],
                                           encoding='utf-8')
    for line in sacct_output.split('\n'):
        if re.search("\s{}\s".format(job_id), line):
            # Status is in the last column : last word in the line.
            status = re.search('\s(\w+)\s*$', line).group(0).strip(' \t\n')
            break
    return status


def _isrunning_canada(path):
    # Read the daemon_log.txt to find all job IDs and check if at least one
    # is active.
    log_path = os.path.join(path, "daemon_log.txt")
    if not os.path.exists(log_path):
        return False
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search("[0-9].*[0-9]$", line)
            if match:
                job_id = match.group(0)
                status = _get_status_canada(job_id)
                if status in ['RUNNING', 'PENDING']:
                    return True
    return False