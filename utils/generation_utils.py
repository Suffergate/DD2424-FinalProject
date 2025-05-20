import torch
import numpy as np


def sample_next_char(preds, temp=1.0, use_nucleus=False, p=0.9):
    """Samples next character from probability distribution"""
    preds = preds.squeeze().detach().cpu().numpy()

    # Apply temperature
    if temp != 1.0:
        preds = np.log(preds) / temp
        preds = np.exp(preds)
        preds = preds / np.sum(preds)

    if use_nucleus:
        # Nucleus sampling (top-p)
        idxs = np.argsort(preds)[::-1]
        probs = preds[idxs]

        # Get cumulative probs
        cum_probs = np.cumsum(probs)

        # Find cutoff
        cutoff = np.where(cum_probs > p)[0][0] + 1

        # Zero out less likely tokens
        probs[cutoff:] = 0

        # Renormalize
        probs = probs / np.sum(probs)

        # Sample
        sample_idx = np.random.choice(len(probs), p=probs)
        return idxs[sample_idx]
    else:
        # Standard sampling
        return np.random.choice(len(preds), p=preds)


def generate_text(
    model,
    seed,
    idx_to_char,
    char_to_idx,
    seq_len,
    num_chars=1000,
    temp=1.0,
    use_nucleus=False,
    p=0.9,
):
    """Generates text using the model"""
    model.eval()

    # Handle seed text
    if len(seed) < seq_len:
        seed = seed.rjust(seq_len)

    if len(seed) > seq_len:
        seed = seed[-seq_len:]

    # Convert to indices
    cur_input = [char_to_idx.get(ch, 0) for ch in seed]
    cur_input = torch.tensor(cur_input, dtype=torch.long).unsqueeze(0).to(model.device)

    gen_text = seed
    hidden = model.init_hidden(1)

    with torch.no_grad():
        for _ in range(num_chars):
            output, hidden = model(cur_input, hidden)

            # Get probs for next char
            probs = torch.softmax(output[:, -1, :], dim=1)

            # Sample
            next_idx = sample_next_char(probs, temp, use_nucleus, p)

            # Add to text
            gen_text += idx_to_char[next_idx]

            # Update input for next step
            cur_input = torch.cat(
                (
                    cur_input[:, 1:],
                    torch.tensor([[next_idx]], dtype=torch.long).to(model.device),
                ),
                dim=1,
            )

    return gen_text
