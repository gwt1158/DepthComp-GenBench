# This code adapts the training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'
# available at https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/main/training/train.py
# which is a modification of the original text-to-image SD training code from the HuggingFace Inc. Team,
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
# licensed under https://github.com/huggingface/diffusers/blob/main/LICENSE

import os
import math
import shutil
import torch
import logging
import transformers
import accelerate
import diffusers
import datasets
from os import makedirs
from os.path import join
from functools import partial
from packaging import version
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from ssd.lr_scheduler import IterExponential
from ssd.model.unetdecmid import UnetDecMid
from ssd.model.depth_autoencoder_mid import DepthAutoencoderMid
from ssd.datasets.mixing import MixedDataLoader
from ssd.preprocessing.constants import (
    RGB,
    DEPTH_RELATIVE,
    GT_RELATIVE, 
    GT_METRIC, 
    GT_METRIC_MASK
)
from ssd.utils import getattr_from_string
from ssd.args import parse_args, save_arguments_to_json, ARGS_JSON

if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def main():
    args = parse_args()

    # Init accelerator and logger
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Save training arguments
    if accelerator.is_main_process:
        makedirs(args.output_dir, exist_ok=True)
        save_arguments_to_json(args, join(args.output_dir, ARGS_JSON))
    if args.noise_type is None:
        logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")

    # Load model components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = DepthAutoencoderMid.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, low_cpu_mem_usage=False, device_map=None)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None)
    combined_model = UnetDecMid(unet, vae)

    # initialize new parts of the model
    combined_model.initialize_fusion(args.initialize_dc)
    
    if args.initialize_dc is not None and not args.skip_vae_weights:
        print("initializing depth encoder from VAE")
        combined_model.initialize_condition_encoder()
    else:
        print("skipping depth encoder initialization from VAE")

    # Freeze or set model components to training mode
    text_encoder.requires_grad_(False)
    combined_model.vae.encoder.requires_grad_(False)
    combined_model.vae.quant_conv.requires_grad_(False)

    combined_model.vae.post_quant_conv.requires_grad_(True)
    combined_model.vae.decoder.requires_grad_(True)
    combined_model.vae.decoder.train()
    combined_model.unet.train()
    
    # Use xformers for efficient attention
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            combined_model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Diffusers model loading and saving functions 
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unetdec"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UnetDecMid.from_pretrained(os.path.join(input_dir, "unetdec"))
                model.unet.register_to_config(**load_model.unet.config)
                model.unet.load_state_dict(load_model.unet.state_dict())
                model.vae.register_to_config(**load_model.vae.config)
                model.vae.load_state_dict(load_model.vae.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        [
            {"params": combined_model.unet.parameters(), "lr": args.learning_rates[0]},
            {"params": combined_model.vae.decoder.parameters(), "lr": args.learning_rates[1]},
            {"params": combined_model.vae.post_quant_conv.parameters(), "lr": args.learning_rates[1]}
        ],
        lr=args.learning_rates[0],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_exp_warmup_steps*accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Training datasets
    hypersim_preprocessing = partial(getattr_from_string(args.train_preprocessing[0]), args)
    hypersim_dataset = getattr_from_string(args.train_dataset[0])(args.train_dataset_path[0], args.train_dataset_subset[0], hypersim_preprocessing)
    hypersim_dataloader = DataLoader(hypersim_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    
    vkitti_preprocessing = partial(getattr_from_string(args.train_preprocessing[1]), args)
    vkitti_dataset = getattr_from_string(args.train_dataset[1])(args.train_dataset_path[1], args.train_dataset_subset[1], vkitti_preprocessing)
    vkitti_dataloader   = DataLoader(vkitti_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    
    train_dataloader = MixedDataLoader(hypersim_dataloader, vkitti_dataloader, split1=args.train_split[0], split2=args.train_split[1])

    # Prepare everything with `accelerator` (Move to GPU)
    combined_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        combined_model, optimizer, train_dataloader, lr_scheduler
    )

    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Calculate number of training steps and epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        wandb_kwargs = { "wandb": { "name": args.wandb_name } }
        accelerator.init_trackers(args.tracker_project_name, tracker_config, init_kwargs=wandb_kwargs)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vkitti_dataset)+len(hypersim_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Resume training from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:  
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps), 
        initial=initial_global_step, 
        desc="Steps", 
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,) 

    loss_fun = getattr_from_string(args.loss)()
    
    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    # Get noise scheduling parameters for later conversion from a parameterized prediction into latent.
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod  = 1 - alpha_prod

    # Training Loop
    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(combined_model):
                # Validity mask
                val_mask = batch[GT_METRIC_MASK].bool().to(device=accelerator.device)

                # End-to-end fine-tuning 
                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if val_mask.any():
                    rgb = batch[RGB].to(device=accelerator.device, dtype=weight_dtype)
                    depth_condition = batch[DEPTH_RELATIVE].to(device=accelerator.device, dtype=weight_dtype)
                    if depth_condition.shape[1] == 3:
                        depth_condition = depth_condition[:, 1:2, :, :]
                    current_estimate = combined_model(rgb, depth_condition, noise_scheduler, empty_encoding, alpha_prod, beta_prod)

                    # Post-process predicted images and retrieve ground truth
                    if args.modality == "dc":
                        current_estimate = current_estimate.mean(dim=1, keepdim=True)
                        current_estimate = torch.clamp(current_estimate, -1, 1)
                        ground_truth = batch[GT_RELATIVE].to(device=accelerator.device, dtype=weight_dtype)
                    elif args.modality == "depth":
                        current_estimate = current_estimate.mean(dim=1, keepdim=True) 
                        current_estimate = torch.clamp(current_estimate,-1,1) 
                        ground_truth = batch[GT_METRIC].to(device=accelerator.device, dtype=weight_dtype)
                    else:
                        raise ValueError(f"Unknown modality {args.modality}")

                    # Loss                    
                    estimation_loss = loss_fun(current_estimate, ground_truth, val_mask)
                    if not torch.isnan(estimation_loss).any():
                        loss = loss + estimation_loss     

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.decoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0
                # Save model checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    logger.info(f"Entered Saving Code at global step {global_step} checkpointing_steps {args.checkpointing_steps}")
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Log loss and learning rate for progress bar
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Break training
            if global_step >= args.max_train_steps:
                break     

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        combined_model = unwrap_model(combined_model)
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
            revision=args.revision, 
            variant=args.variant
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=combined_model.vae,
            unet=combined_model.unet,
            scheduler=scheduler,
            revision=args.revision,
            variant=args.variant,
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)

    logger.info(f"Finished training.")
    accelerator.end_training()

if __name__ == "__main__":
    main()