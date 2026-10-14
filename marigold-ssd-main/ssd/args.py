import os
import json
import argparse
import datetime
from ssd.utils import load_json_to_dict

ARGS_JSON = "arguments.json"

OPT_INTERPOLATION = "interpolation"
OPT_VAE_MID = "vae_mid"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, choices=["depth", "normals", "dc"], required=True)
    parser.add_argument("--noise_type", type=str, default=None, choices=["zeros", "gaussian", "pyramid"],
        help="If left as None, Stable Diffusion checkpoints can be trained without altering the input channels (i.e., only 4 input channels for RGB input).")
    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=20000)
    # Stable diffusion training settings
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--output_dir", type=str, default="training/model-finetuned", required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--max_train_steps", type=int, default=None, required=True, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--learning_rates", nargs="+", type=float, default=None, help="List of learning rates for training.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type=int, default=20000,help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or \"latest\" to automatically select the last available checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    parser.add_argument("--train_dataset", nargs="+", help="Dataset classes.")
    parser.add_argument("--train_dataset_subset", nargs="+", help="Subsets of dataset used for training.")
    parser.add_argument("--train_dataset_path", nargs="+", default=None, help="Paths to load datasets from.")
    parser.add_argument("--train_split", nargs="+", type=float, default=None, help="Portion of datasets used for training.")
    parser.add_argument("--preprocessing_paths", nargs="+", default=None, help="Path where additional data should be loaded by preprocessing.")
    parser.add_argument("--train_preprocessing", nargs="+", default=None)
    parser.add_argument("--loss", type=str, default=None, help="Class of loss to be used for training.")
    parser.add_argument("--naive_lidar", type=float, default=None)
    parser.add_argument("--naive_lidar_max", type=float, default=None)

    parser.add_argument("--initialize_dc", type=str, default=None, help="Initialize DC related layers.")
    parser.add_argument("--skip_vae_weights", action="store_true", 
                        help="If provided, sparse depth encoder will keep torch default initialization instead of copying weights from VAE.")

    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--tracker_project_name", type=str, default="dc2")
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--unet_subfolder", type=str, default=None)
    parser.add_argument("--vae_subfolder", type=str, default=None)
    parser.add_argument("--model_vae", type=str, default=None, help="Load other VAE than the one stored in --model")
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--preprocessing_paths", nargs="+", default=None)
    parser.add_argument("--eval_dataset_subset", type=str, default=None)
    parser.add_argument("--eval_preprocessing", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--half_precision", "--fp16", action="store_true", help="Run with half-precision (16-bit float), might lead to suboptimal result.")
    parser.add_argument("--noise", type=str, default='zeros', choices=["gaussian", "pyramid", "zeros"])
    parser.add_argument("--timestep_spacing", type=str, default='trailing', choices=["trailing", "leading"])
    parser.add_argument("--model_type", type=str, choices=[OPT_INTERPOLATION, OPT_VAE_MID]) 
    parser.add_argument("--naive_lidar", type=float, default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--visualize_depth", action="store_true")
    parser.add_argument("--limit_dataset_size", type=int, default=None)
    return parser.parse_args()


def parse_eval_args(json_path: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Directory of depth predictions")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--preprocessing_paths", nargs="+", default=None)
    parser.add_argument("--postprocessing", type=str, default=None)
    parser.add_argument("--eval_dataset_subset", type=str, default=None)
    parser.add_argument("--limit_dataset_size", type=int, default=None)
    parser.add_argument("--eval_preprocessing", type=str, default=None)
    parser.add_argument("--alignment", choices=[None, "least_square", "least_square_disparity"], default=None, help="Method to estimate scale and shift between predictions and ground truth.")
    parser.add_argument("--align_with", choices=[None, "groundtruth", "depth"], default=None)
    parser.add_argument("--alignment_max_res", type=int, default=None, help="Max operating resolution used for LS alignment")
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")
    parser.add_argument("--naive_lidar", type=float, default=None)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--visualize_error", action="store_true")

    if json_path is not None:
        default_args = vars(parser.parse_args([]))
        default_args.update(load_json_to_dict(json_path))
        return argparse.Namespace(**default_args)

    return parser.parse_args()


def save_arguments_to_json(args: argparse.Namespace, path: str, timestamp: bool = True):
    argparse_dict = vars(args)

    if timestamp:
        argparse_dict["timestamp"] = str(datetime.datetime.now())

    with open(path, "w") as file:
        file.write(json.dumps(argparse_dict, indent=2, sort_keys=True))


def load_args_from_json(path: str) -> argparse.Namespace:
    with open(path, 'r') as json_file:
        args_dict = json.load(json_file)
        return argparse.Namespace(**args_dict)
