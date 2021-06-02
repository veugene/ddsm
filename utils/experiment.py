from collections import OrderedDict
import imp
import os
import sys

from natsort import natsorted
import torch
import numpy as np
import json
from ignite.engine import (Engine,
                           Events)
from ignite.handlers import ModelCheckpoint

from .radam import RAdam
from .trackers import progress_report


class experiment(object):
    """
    Parse and set up arguments, set up the model and optimizer, and prepare 
    the folder into which to save the experiment.

    In args, expecting `model_from` and `path`.
    
    score_function : for tracking best checkpoint.
    """
    def __init__(self, args, score_function=None):
        self.args = args
        self.epoch = 0
        self.experiment_path = args.path
        self.model = None
        self.optimizer = None
        
        # Make experiment directory, if necessary.
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        # Get model config.
        # 
        # NOTE: assuming that this always exists. Currently, dispatch
        # reloads all initial arguments on resume, so `model_from` is
        # defined even on resume. Taking `model_from` always from args
        # instead of from saved state allows it to be overriden on resume
        # by passing the argument anew.
        # 
        # NOTE: the model config file has to exist in its original location
        # when resuming an experiment.
        with open(args.model_from) as f:
            self.model_as_str = f.read()

        # Initialize model and optimizer.
        self._init_state(optimizer_name=args.optimizer,
                         learning_rate=args.learning_rate,
                         opt_kwargs=args.opt_kwargs,
                         weight_decay=args.weight_decay,
                         model_kwargs=args.model_kwargs,
                         score_function=score_function)
            
        # Does the experiment directory already contain state files?
        state_file_list = natsorted([fn for fn in os.listdir(args.path)
                                     if fn.startswith('state_checkpoint_')
                                     and fn.endswith('.pt')])
        
        # If yes, resume; else, initialize a new experiment.
        if os.path.exists(args.path) and len(state_file_list):
            # RESUME old experiment
            state_file = state_file_list[-1]
            state_from = os.path.join(args.path, state_file)
            print(f"Resuming from {state_from}.")
            state_dict = torch.load(state_from)
            self.load_state_dict(state_dict)
        else:
            # INIT new experiment
            with open(os.path.join(args.path, "args.txt"), 'w') as f:
                f.write('\n'.join(sys.argv))
            if args.weights_from is not None:
                # Load weights from specified checkpoint.
                self.load_model(load_from=args.weights_from)
            with open(os.path.join(args.path, "config.py"), 'w') as f:
                f.write(self.model_as_str)
        
        # Initialization complete.
        print("Number of parameters\n"+
              "\n".join([f" {key} : {count_params(self.model[key])}"
                         for key in self.model.keys()
                         if hasattr(self.model[key], 'parameters')]))
    
    def create_engine(self, function,
                     append=True, prefix=None, epoch_length=None):
        engine = Engine(function)
        fn = "log.txt" if prefix is None else f"{prefix}_log.txt"
        progress = progress_report(
            prefix=prefix,
            append=append,
            epoch_length=epoch_length,
            log_path=os.path.join(self.experiment_path, fn))
        progress.attach(engine)
        return engine
    
    """
    Attach to `trainer` (training) and/or `evaluator` (validation) engine.
    This attaches checkpoint events to the engines and sets up epoch tracking
    with the training engine.
    """
    def attach(self, trainer, evaluator=None):
        # Track epoch in the experiment object, the training engine state,
        # and the checkpoint handlers.
        # 
        # This allows the correct epoch to be reported, training to stop after
        # the correct number of epochs, resumed experiments to start on the
        # correct epoch, and saved states to be enumerated with the correct
        # epoch.
        trainer.add_event_handler(Events.STARTED,
                                  lambda engine: setattr(engine.state,
                                                         "epoch",
                                                         self.epoch))
        trainer.add_event_handler(Events.EPOCH_STARTED,
                                  lambda engine: setattr(self,
                                                         "epoch",
                                                         engine.state.epoch))
        
        # Set up calls to checkpoint handlers.
        to_save = {'checkpoint': self}
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda engine:self.checkpoint_last_handler(engine, to_save))
        if evaluator is not None and self.checkpoint_best_handler is not None:
            evaluator.add_event_handler(
                Events.EPOCH_COMPLETED,
                lambda engine:self.checkpoint_best_handler(engine, to_save))
    
    def get_epoch(self):
        return self.epoch
    
    def _init_state(self, optimizer_name, learning_rate=0.,
                    opt_kwargs=None, weight_decay=0., model_kwargs=None,
                    score_function=None):
        '''
        Initialize the model, its state, and the optimizer's state.
        
        Requires the model to be defined in `self.model_as_str`.
        '''
        
        # Build the model.
        if model_kwargs is None:
            model_kwargs = {}
        module = imp.new_module('module')
        exec(self.model_as_str, module.__dict__)
        model = getattr(module, 'build_model')(**model_kwargs)
        if not isinstance(model, dict):
            model = {'model': model}
        
        # If optimizer_name is in JSON format, convert string to dict.
        try:
            optimizer_name = json.loads(optimizer_name)
        except ValueError:
            # Not in JSON format; keep as a string.
            optimizer_name = optimizer_name
        
        # Setup the optimizer.
        optimizer = {}
        for key in model.keys():
            if not hasattr(model[key], 'parameters'):
                continue    # Model has no parameters and cannot be optimized.
            def parse(arg):
                # Helper function for args when passed as dict, with
                # model names as keys.
                if isinstance(arg, dict) and key in arg:
                    return arg[key]
                return arg
            model[key].cuda()
            optimizer[key] = self._get_optimizer(
                                            name=parse(optimizer_name),
                                            params=model[key].parameters(),
                                            lr=parse(learning_rate),
                                            opt_kwargs=parse(opt_kwargs),
                                            weight_decay=weight_decay)
        
        # Store model, optimizer.
        self.model = model
        self.optimizer = optimizer
        
        # Function to return epoch number to checkpoint handler.
        def epoch_step(engine, event_name):
            return self.epoch
        
        # Checkpoint at every epoch.
        checkpoint_last_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='state',
                                    n_saved=2,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False,
                                    global_step_transform=epoch_step)
        
        # Checkpoint for best model performance.
        checkpoint_best_handler = None
        if score_function is not None:
            checkpoint_best_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='best_state',
                                    n_saved=1,
                                    score_function=score_function,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False,
                                    global_step_transform=epoch_step)
        
        # Store checkpoint handlers.
        self.checkpoint_last_handler = checkpoint_last_handler
        self.checkpoint_best_handler = checkpoint_best_handler

    def load_model(self, path):
        state_dict = torch.load(path)
        for key in state_dict:
            if key.startswith('model_'):
                m_key = key.replace('model_', '')
                self.model[m_key].load_state_dict(state_dict[f'model_{m_key}'])
    
    def load_last_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('state_checkpoint_')
                                and fn.endswith('.pt')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        state_dict = torch.load(state_from)
        self.load_state_dict(state_dict)
    
    def load_best_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('best_state_checkpoint_')
                                and fn.endswith('.pt')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        state_dict = torch.load(state_from)
        self.load_state_dict(state_dict)

    def _get_optimizer(self, name, params, lr=0., opt_kwargs=None, 
                       weight_decay=0.):
        kwargs = {'params'       : [p for p in params if p.requires_grad],
                  'lr'           : lr,
                  'weight_decay' : weight_decay}
        optimizer = None
        if name=='adam' or name=='amsgrad':
            if opt_kwargs is None:
                opt_kwargs = {'betas': (0.5, 0.999)}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.Adam(amsgrad=bool(name=='amsgrad'),
                                         **kwargs)
        elif name=='radam':
            if opt_kwargs is None:
                opt_kwargs = {'betas': (0.5, 0.999)}
            kwargs.update(opt_kwargs)
            optimizer = RAdam(**kwargs)
        elif name=='rmsprop':
            if opt_kwargs is None:
                opt_kwargs = {'alpha': 0.5}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.RMSprop(**kwargs)
        elif name=='sgd':
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise ValueError(f"Optimizer {name} not supported.")
        return optimizer
    
    def state_dict(self):
        state_dict = OrderedDict([('model_as_str', self.model_as_str),
                                  ('epoch', self.epoch)])
        for key in sorted(self.model.keys()):
            state_dict[f'model_{key}'] = self.model[key].state_dict()
            state_dict[f'optimizer_{key}'] = self.optimizer[key].state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        # Check if the model config was the same as what is initialized now.
        # Do not use the saved config; stick with the new config. Sometimes,
        # it is useful to change the model logic without changing the
        # parameters. Also, there's no reason to assume the saved config
        # is more correct than the actual config used to create the actual
        # model for which parameters are loaded below.
        if state_dict['model_as_str'] != self.model_as_str:
            print("NOTE : model configuration differs from the one used with "
                  "the last saved state. Using the new configuration.")
        
        # Load epoch.
        self.epoch = state_dict['epoch']
        
        # Load models and optimizers.
        for key in state_dict:
            if key.startswith('model_') and key != 'model_as_str':
                m_key = key.replace('model_', '')
                self.model[m_key].load_state_dict(
                    state_dict[f'model_{m_key}'])
                self.optimizer[m_key].load_state_dict(
                    state_dict[f'optimizer_{m_key}'])


def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num 