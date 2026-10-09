import torch


def run_mc_classic(
    model,
    query,
    noise_sigma=0.05,
    number_iterations=1000,
    threshold=0.5,
    clip_range=None,
):
    """
    Executes conventional Monte Carlo simulation by generating a single noisy input sequence
    per iteration inside a loop and evaluating it through the full CNN.
    """
    model.eval()
    number_class_0 = 0
    number_class_1 = 0
    sigmoid_sum = 0.0

    with torch.no_grad():
        for _ in range(number_iterations):
            # Generate a single noisy input sequence per iteration
            gaussian_noise = noise_sigma * torch.randn_like(query)
            noisy_query = query + gaussian_noise

            if clip_range is not None:
                lower_bound, upper_bound = clip_range
                noisy_query = torch.clamp(noisy_query, min=lower_bound, max=upper_bound)

            # Process single noisy sequence through the entire CNN model
            sigmoid_value = model(noisy_query).reshape(-1)[0].item()
            sigmoid_sum += sigmoid_value

            if sigmoid_value < threshold:
                number_class_0 += 1
            else:
                number_class_1 += 1

    return {
        "number_iterations": number_iterations,
        "number_class_0": number_class_0,
        "number_class_1": number_class_1,
        "relative_frequency_class_0": number_class_0 / number_iterations,
        "relative_frequency_class_1": number_class_1 / number_iterations,
        "mean_sigmoid": sigmoid_sum / number_iterations,
    }