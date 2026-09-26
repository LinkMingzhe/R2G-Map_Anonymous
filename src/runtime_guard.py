def validate_config(config, inference=False):
    stage = "stage2" if inference else str(config.training.stage)
    if stage == "stage2" and not bool(config.model.get("use_timestep_condition", False)):
        raise ValueError("This release uses TSP Stage2: use_timestep_condition must be true.")
