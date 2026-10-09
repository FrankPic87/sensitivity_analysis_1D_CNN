import math
import torch


def run_linearization(model, query, noise_sigma):
    """
    Analytically evaluates the output probability distribution of the linearized model
    via first-order Taylor expansion and Probit approximation.
    """
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma cannot be negative.")

    model.eval()

    x = query.detach().clone()
    x.requires_grad_(True)

    # Forward pass to calculate nominal logit f(x)
    activations = model.relu(model.conv(x))
    activations_flat = torch.flatten(activations, start_dim=1)
    logit = model.fc(activations_flat).reshape(-1)[0]

    # Gradient of the logit with respect to the input sequence
    input_gradient = torch.autograd.grad(
        outputs=logit,
        inputs=x,
        create_graph=False,
        retain_graph=False,
    )[0]

    mean_logit = logit.item()
    gradient_sq_norm = torch.sum(input_gradient ** 2).item()
    logit_variance = (noise_sigma ** 2) * gradient_sq_norm

    if logit_variance > 0.0:
        logit_std = math.sqrt(logit_variance)

        # Theoretical probability of Class 1 assignment: P(f(x + eps) > 0)
        z_score = mean_logit / logit_std
        prob_class_1 = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))

        # Expected mean sigmoid output via Probit approximation
        probit_scale = math.sqrt(1.0 + (math.pi / 8.0) * logit_variance)
        expected_sigmoid = 1.0 / (1.0 + math.exp(-mean_logit / probit_scale))
    else:
        prob_class_1 = 1.0 if mean_logit >= 0.0 else 0.0
        expected_sigmoid = 1.0 / (1.0 + math.exp(-mean_logit))

    prob_class_0 = 1.0 - prob_class_1

    return {
        "mean_logit": mean_logit,
        "logit_variance": logit_variance,
        "logit_std": math.sqrt(logit_variance) if logit_variance > 0.0 else 0.0,
        "relative_frequency_class_0": prob_class_0,
        "relative_frequency_class_1": prob_class_1,
        "expected_sigmoid": expected_sigmoid,
    }