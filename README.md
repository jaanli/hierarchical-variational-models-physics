# hierarchical-variational-models-physics
Hierarchical variational models for physics.

## Running an experiment

To fit an Ising model with 1M+ spins using 5400 parameters:

```
python main.py --seed=58283 --model=ising --boundary=periodic --max_iteration=1000000000 --use_gpu=True --num_samples_grad=8 --flow_depth=6 --activation=relu --num_samples_print=256 --variational_posterior=RealNVPPosterior --prior_std=1.0 --posterior_std=1.0 --control_variate=False --rao_blackwellize=True --marginalize=False --learning_rate=1e-05 --momentum=0.9 --log_interval=10 --beta=0.4 --flow_type=realnvp --hidden_size=8 --print_batch_size=128 --num_spins=1048576 --log_dir=/tmp
```
