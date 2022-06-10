# Example 1: Computer Vision

The [1_cv_example.py](./1_cv_example.py) script is a simple example to fine-tune a ResNet-50 on a classification task ([Ofxord-IIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)).

The same script can be run in any of the following configurations:
- single CPU or single GPU
- multi GPUs (using PyTorch distributed mode)
- (multi) TPUs
- fp16 (mixed-precision) or fp32 (normal precision)

Prior to running it you should install the relevant packages:

Optional:
```bash
virtualenv env
. env/bin/activate
```

```bash
pip install -r requirements.txt
```

and you should download the data with the following commands:

```bash
mkdir data
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xzf images.tar.gz
```

To enable profiling and tracing the performance, run the same command but wrap the script with the interface.

For example, TBD ...
```

```

To run it in each of these various modes, use the following commands:
- single CPU:
    * from a server without GPU
        ```bash
        python ./1_cv_example.py --data_dir path_to_data
        ```
    * from any server by passing `cpu=True` to the `Accelerator`.
        ```bash
        python ./1_cv_example.py --data_dir path_to_data --cpu
        ```
    * from any server with Accelerate launcher
        ```bash
        accelerate launch --cpu ./1_cv_example.py --data_dir path_to_data
        ```
- single GPU:
    ```bash
    python ./1_cv_example.py  # from a server with a GPU
    ```
- multi GPUs (using PyTorch distributed mode)
    * With Accelerate config and launcher
        ```bash
        accelerate config  # This will create a config file on your server
        accelerate launch ./1_cv_example.py --data_dir path_to_data  # This will run the script on your server
        ```
    * With traditional PyTorch launcher
        ```bash
        python -m torch.distributed.launch --nproc_per_node 2 --use_env ./1_cv_example.py --data_dir path_to_data
        ```
- multi GPUs, multi node (several machines, using PyTorch distributed mode)
    * With Accelerate config and launcher, on each machine:
        ```bash
        accelerate config  # This will create a config file on each server
        accelerate launch ./1_cv_example.py --data_dir path_to_data  # This will run the script on each server
        ```
    * With PyTorch launcher only
        ```bash
        python -m torch.distributed.launch --nproc_per_node 2 \
            --use_env \
            --node_rank 0 \
            --master_addr master_node_ip_address \
            ./cv_example.py --data_dir path_to_data  # On the first server
        python -m torch.distributed.launch --nproc_per_node 2 \
            --use_env \
            --node_rank 1 \
            --master_addr master_node_ip_address \
            ./cv_example.py --data_dir path_to_data  # On the second server
        ```
