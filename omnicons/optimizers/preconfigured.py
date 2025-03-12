def get_adamw(model_params, learning_rate):
    from transformers import AdamW

    return AdamW(model_params, lr=learning_rate)


def get_deepspeed_adamw(model_params, learning_rate):
    """Deep Speed's Implementation of ADAMW (CPU)
    DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
    In order to apply this optimizer, the model requires to have its master parameter
    (in FP32) reside on the CPU memory.
    See https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu for details
    """
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    return DeepSpeedCPUAdam(
        model_params=model_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        adamw_mode=True,
    )


def get_deepspeed_adam(model_params, learning_rate):
    """Deep Speed's Implementation of ADAM (CPU)
    DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
    In order to apply this optimizer, the model requires to have its master parameter
    (in FP32) reside on the CPU memory.
    See https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu for details
    """
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    return DeepSpeedCPUAdam(
        model_params=model_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        adamw_mode=False,
    )
