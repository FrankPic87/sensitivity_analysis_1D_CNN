import torch


def equivalent_dropout_from_gaussian_noise(model, query, noise_sigma):
    """
    Calculates the dropout probability p that locally yields the same logit variance
    as the input Gaussian measurement noise (sigma).
    """
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma cannot be negative.")

    model.eval()
    x = query.detach().clone()
    x.requires_grad_(True)

    activations = model.relu(model.conv(x))
    activations_flat = torch.flatten(activations, start_dim=1)
    logit = model.fc(activations_flat).reshape(-1)[0]

    input_gradient = torch.autograd.grad(
        outputs=logit,
        inputs=x,
        create_graph=False,
        retain_graph=False,
    )[0]

    fc_weights = model.fc.weight.reshape(-1)
    activation_values = activations_flat.reshape(-1)

    dropout_factor = torch.sum((fc_weights * activation_values) ** 2)
    gaussian_factor = torch.sum(input_gradient ** 2)
    gaussian_logit_variance = (noise_sigma ** 2) * gaussian_factor

    denominator = dropout_factor + gaussian_logit_variance

    if denominator.item() <= 0.0:
        raise ValueError("Zero local sensitivity for the given query sequence.")

    dropout_probability = (gaussian_logit_variance / denominator).item()

    if not (0.0 <= dropout_probability < 1.0):
        raise ValueError("Calculated dropout probability is out of bounds [0, 1).")

    return dropout_probability


def run_mc_dropout(model, query, noise_sigma, number_iterations=1000, threshold=0.5):
    """
    Executes MC Dropout by pre-computing the feature extraction layers (Conv1D + ReLU) ONCE,
    and repeating ONLY the Dropout mask generation and FC classification head for each iteration.
    """
    # 1. Compute equivalent dropout probability and update model parameter
    dropout_p = equivalent_dropout_from_gaussian_noise(model, query, noise_sigma)
    model.dropout.p = dropout_p

    # 2. Pre-compute deterministic representations (Conv1D + ReLU) ONCE
    model.eval()
    with torch.no_grad():
        pre_dropout_activations = model.relu(model.conv(query))

    # 3. Enable training mode exclusively on the dropout layer
    model.dropout.train()

    number_class_0 = 0
    number_class_1 = 0
    sigmoid_sum = 0.0

    # 4. Loop across iterations evaluating ONLY Dropout -> FC -> Sigmoid
    with torch.no_grad():
        for _ in range(number_iterations):
            x = model.dropout(pre_dropout_activations)
            x = torch.flatten(x, start_dim=1)
            logit = model.fc(x)
            sigmoid_val = torch.sigmoid(logit).reshape(-1)[0].item()

            sigmoid_sum += sigmoid_val
            if sigmoid_val < threshold:
                number_class_0 += 1
            else:
                number_class_1 += 1

    return {
        "dropout_probability": dropout_p,
        "number_iterations": number_iterations,
        "number_class_0": number_class_0,
        "number_class_1": number_class_1,
        "relative_frequency_class_0": number_class_0 / number_iterations,
        "relative_frequency_class_1": number_class_1 / number_iterations,
        "mean_sigmoid": sigmoid_sum / number_iterations,
    }