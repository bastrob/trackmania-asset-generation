import torch


@torch.no_grad()
def sample(
    model,
    scheduler,
    device,
    image_size=(3, 512, 512),
    num_inference_steps=1000
):
    model.eval()

    # Start from pure noise
    image = torch.randn((1, *image_size), device=device)

    scheduler.set_timesteps(num_inference_steps)

    for timestep in scheduler.timesteps:
        t = torch.tensor([timestep], device=device)

        # Predict noise
        noise_pred = model(image, t)

        # Reverse diffusion step
        image = scheduler.step(
            noise_pred,
            timestep,
            image
        ).prev_sample
    
    return image