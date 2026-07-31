import torch


def load_qsfusion_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    message = model.load_state_dict(state_dict, strict=False)
    return checkpoint, message
