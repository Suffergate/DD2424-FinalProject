import torch
import torch.nn as nn
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from .generation_utils import generate_text


def validate(model, dataloader, criterion, device):
    """Evaluate model on data"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize hidden state
            hidden = model.init_hidden(batch_size)

            # Forward pass
            outputs, _ = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens

    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path=None,
    plot_data_path=None,
    scheduler=None,
    validate_every=1000,
    generate_every=5000,
    char_to_idx=None,
    idx_to_char=None,
    raw_text=None,
    sample_dir=None,
):
    """Train model and save text samples at regular intervals"""
    # Track metrics
    train_losses = []
    val_losses = []
    val_iterations = []
    generated_samples = []

    best_val_loss = float("inf")
    iteration = 0

    # Epoch-level statistics
    epoch_train_losses = []
    epoch_val_losses = []

    # Get model name and create progression directory
    model_name = type(model).__name__
    progression_dir = os.path.join(sample_dir, "progression")
    os.makedirs(progression_dir, exist_ok=True)

    # Create a summary file to track all generated samples
    summary_file = os.path.join(progression_dir, "generation_progress.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Text Generation Progress for {model_name}\n\n")
        f.write("This file tracks how text generation improves during training.\n\n")
        f.write("| Iteration | Sample Link | Sampling Method |\n")
        f.write("|-----------|-------------|----------------|\n")

    # Save initial sample
    if char_to_idx and idx_to_char and raw_text and sample_dir:
        seed_text = raw_text[:100]

        # Generate with standard sampling
        sample_filename = f"{model_name}_iter0_standard.txt"
        sample_path = os.path.join(progression_dir, sample_filename)

        with torch.no_grad():
            try:
                generated_text = generate_text(
                    model,
                    seed_text,
                    idx_to_char,
                    char_to_idx,
                    len(seed_text),
                    num_chars=500,
                    temperature=1.0,
                )
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write(f"Seed text: {seed_text[:50]}\n\n")
                    f.write(f"Generated text (iteration 0):\n\n")
                    f.write(generated_text)

                # Update summary file
                with open(summary_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"| 0 | [Standard Sample](./progression/{sample_filename}) | Standard (temp=1.0) |\n"
                    )
            except Exception as e:
                print(f"Error generating initial sample: {str(e)}")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train for one epoch
        model.train()
        epoch_train_loss = 0
        epoch_total_tokens = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize hidden state for this batch
            hidden = model.init_hidden(batch_size)

            # Detach hidden state
            if isinstance(hidden, tuple):
                hidden = tuple(h.detach() for h in hidden)
            else:
                hidden = hidden.detach()

            optimizer.zero_grad()

            # Forward pass
            outputs, hidden = model(inputs, hidden)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1)

            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            batch_loss = loss.item()
            num_tokens = targets.size(0)

            # Track loss by tokens
            epoch_train_loss += batch_loss * num_tokens
            epoch_total_tokens += num_tokens

            # Track batch-level metrics
            train_losses.append(batch_loss)

            iteration += 1

            # Generate text samples at regular intervals
            if (
                char_to_idx
                and idx_to_char
                and raw_text
                and sample_dir
                and iteration % generate_every == 0
            ):
                seed_text = raw_text[:100]

                # Generate with standard sampling
                sample_filename = f"{model_name}_iter{iteration}_standard.txt"
                sample_path = os.path.join(progression_dir, sample_filename)

                with torch.no_grad():
                    try:
                        generated_text = generate_text(
                            model,
                            seed_text,
                            idx_to_char,
                            char_to_idx,
                            len(seed_text),
                            num_chars=500,
                            temperature=1.0,
                        )
                        with open(sample_path, "w", encoding="utf-8") as f:
                            f.write(f"Seed text: {seed_text[:50]}\n\n")
                            f.write(f"Generated text (iteration {iteration}):\n\n")
                            f.write(generated_text)

                        # Record sample metadata
                        generated_samples.append(
                            {
                                "iteration": iteration,
                                "filename": sample_filename,
                                "sampling": "standard",
                                "temperature": 1.0,
                            }
                        )

                        # Update summary file
                        with open(summary_file, "a", encoding="utf-8") as f:
                            f.write(
                                f"| {iteration} | [Standard Sample](./progression/{sample_filename}) | Standard (temp=1.0) |\n"
                            )
                    except Exception as e:
                        print(
                            f"Error generating sample at iteration {iteration}: {str(e)}"
                        )

                # Generate with nucleus sampling
                sample_filename = f"{model_name}_iter{iteration}_nucleus.txt"
                sample_path = os.path.join(progression_dir, sample_filename)

                with torch.no_grad():
                    try:
                        generated_text = generate_text(
                            model,
                            seed_text,
                            idx_to_char,
                            char_to_idx,
                            len(seed_text),
                            num_chars=500,
                            temperature=1.0,
                            use_nucleus=True,
                            p=0.9,
                        )
                        with open(sample_path, "w", encoding="utf-8") as f:
                            f.write(f"Seed text: {seed_text[:50]}\n\n")
                            f.write(
                                f"Generated text (iteration {iteration}, nucleus sampling):\n\n"
                            )
                            f.write(generated_text)

                        # Record sample metadata
                        generated_samples.append(
                            {
                                "iteration": iteration,
                                "filename": sample_filename,
                                "sampling": "nucleus",
                                "p": 0.9,
                            }
                        )

                        # Update summary file
                        with open(summary_file, "a", encoding="utf-8") as f:
                            f.write(
                                f"| {iteration} | [Nucleus Sample](./progression/{sample_filename}) | Nucleus (p=0.9) |\n"
                            )
                    except Exception as e:
                        print(
                            f"Error generating nucleus sample at iteration {iteration}: {str(e)}"
                        )

                print(f"\nGenerated text samples at iteration {iteration}")

            # Validate periodically
            if iteration % validate_every == 0 or iteration == 100:
                val_loss = validate(
                    model, val_loader, criterion, device
                )

                val_losses.append(val_loss)
                val_iterations.append(iteration)

                print(
                    f"Iteration {iteration} | Val Loss: {val_loss:.4f}"
                )

                # Save model if best
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "iteration": iteration,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": epoch_train_loss / epoch_total_tokens,
                            "val_loss": val_loss,
                        },
                        save_path,
                    )
                    print(f"Model saved to {save_path} (new best)")

                model.train()

        # Calculate epoch-level statistics
        epoch_avg_train_loss = epoch_train_loss / epoch_total_tokens

        epoch_train_losses.append(epoch_avg_train_loss)

        # Final validation for this epoch
        val_loss = validate(model, val_loader, criterion, device)

        epoch_val_losses.append(val_loss)

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    # Generate a final sample
    if char_to_idx and idx_to_char and raw_text and sample_dir:
        seed_text = raw_text[:100]

        # Generate with standard sampling
        sample_filename = f"{model_name}_final_standard.txt"
        sample_path = os.path.join(progression_dir, sample_filename)

        with torch.no_grad():
            try:
                generated_text = generate_text(
                    model,
                    seed_text,
                    idx_to_char,
                    char_to_idx,
                    len(seed_text),
                    num_chars=500,
                    temperature=1.0,
                )
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write(f"Seed text: {seed_text[:50]}\n\n")
                    f.write(f"Generated text (final):\n\n")
                    f.write(generated_text)

                # Update summary file
                with open(summary_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"| Final | [Standard Sample](./progression/{sample_filename}) | Standard (temp=1.0) |\n"
                    )
            except Exception as e:
                print(f"Error generating final sample: {str(e)}")

    # Save plotting data
    if plot_data_path:
        plot_data = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_iterations": val_iterations,
            "iterations": list(range(1, len(train_losses) + 1)),
            "epoch_train_losses": epoch_train_losses,
            "epoch_val_losses": epoch_val_losses,
            "generated_samples": generated_samples,
        }

        with open(plot_data_path, "wb") as f:
            pickle.dump(plot_data, f)
        print(f"Plotting data saved to {plot_data_path}")

    return (
        train_losses,
        val_losses,
        val_iterations,
    )


def plot_loss(
    train_losses,
    val_losses,
    val_iterations,
    title="Model Training Metrics",
    save_path=None,
):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)

    # Plot loss on the first axis
    iterations = np.arange(1, len(train_losses) + 1)

    # Smoothing training loss
    window_size = min(100, len(train_losses) // 10)
    if window_size > 1:
        train_smooth = np.convolve(
            train_losses, np.ones(window_size) / window_size, mode="valid"
        )
        smooth_iterations = iterations[window_size - 1 :]
        ax1.plot(
            smooth_iterations,
            train_smooth,
            "b-",
            label="Training Loss (Smoothed)",
            linewidth=2,
        )
    else:
        ax1.plot(iterations, train_losses, "b-", label="Training Loss", alpha=0.6)

    # Plot validation loss
    ax1.plot(
        val_iterations,
        val_losses,
        "ro-",
        label="Validation Loss",
        markersize=4,
        linewidth=2,
    )

    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Iterations (Batches)")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Metrics plot saved to {save_path}")

    return fig


def calculate_test_accuracy(model, test_loader, device):
    """Calculate accuracy on test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(batch_size)

            outputs, _ = model(inputs, hidden)

            # Get predictions
            _, predicted = torch.max(outputs, 2)

            # Calculate accuracy
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    return 100 * correct / total