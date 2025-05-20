import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import pickle
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Imports
from config import *
from utils.training_utils import train_model, validate
from utils.data_utils import RandomTextDataset, get_raw_text
from Models.lstm_2_layer import LSTM2Model


def run_grid_search():
    """Run grid search to find optimal lr and batch size"""
    print("\n" + "=" * 50)
    print("STARTING GRID SEARCH")
    print("=" * 50)

    # Grid params
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [32, 64, 128]

    # Make dir for results
    grid_dir = os.path.join(MODEL_SAVE_DIR, "grid_search")
    os.makedirs(grid_dir, exist_ok=True)

    # Results tracking
    results = []

    # Load data
    print("Loading data")
    with open(PROCESSED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    char_to_idx = data["char_to_idx"]
    idx_to_char = data["idx_to_char"]
    vocab_size = data["vocab_size"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]

    # Get text for generation
    raw_text = get_raw_text()

    # Fixed params
    h_dim = HIDDEN_DIM
    emb_dim = EMBEDDING_DIM
    dropout = DROPOUT
    seq_len = SEQ_LENGTH
    n_epochs = NUM_EPOCHS 

    # Results file
    results_file = os.path.join(grid_dir, "grid_search_results.txt")
    with open(results_file, "w") as f:
        f.write("LR | Batch Size | Val Loss | Time (s)\n")
        f.write("-" * 100 + "\n")

    # Best tracking
    best_val_loss = float("inf")
    best_params = None
    best_model_path = None

    # Start timer
    grid_start = time.time()

    # Run search
    for lr in lrs:
        for bs in batch_sizes:
            print(f"\n{'-'*50}")
            print(f"Testing lr={lr}, bs={bs}")
            print(f"{'-'*50}")

            # Create model name and paths
            model_name = f"lstm2_lr{lr}_bs{bs}"
            model_path = os.path.join(grid_dir, f"{model_name}_model.pt")
            main_model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_model.pt")
            plot_data_path = os.path.join(grid_dir, f"{model_name}_plot_data.pkl")
            plot_path = os.path.join(grid_dir, f"{model_name}_loss.png")

            # Check if model already exists in main model directory
            if os.path.exists(main_model_path):
                print(f"Found existing model at {main_model_path}, loading")
                try:
                    # Load the model to get validation loss
                    checkpoint = torch.load(main_model_path, map_location=DEVICE)
                    model_best_val_loss = checkpoint.get("val_loss", float("inf"))
                    training_time = checkpoint.get("training_time", 0)
                    best_iter = checkpoint.get("iteration", 0)

                    # Save results
                    result = {
                        "lr": lr,
                        "bs": bs,
                        "val_loss": model_best_val_loss,
                        "best_iter": best_iter,
                        "time": training_time,
                        "model_path": main_model_path,
                    }
                    results.append(result)

                    # Update results file
                    with open(results_file, "a") as f:
                        f.write(
                            f"{lr:<12} | {bs:<10} | {model_best_val_loss:<8.4f} | {training_time:<15.2f} | (loaded from existing model)\n"
                        )

                    # Check if best
                    if model_best_val_loss < best_val_loss:
                        best_val_loss = model_best_val_loss
                        best_params = (lr, bs)
                        best_model_path = main_model_path

                    print(
                        f"Using existing model with val loss: {model_best_val_loss:.4f}"
                    )
                    continue  # Skip to next parameter combination
                except Exception as e:
                    print(f"Error loading existing model: {e}")
                    print("Will train a new model instead.")

            # Create loaders with this batch size
            train_ds = RandomTextDataset(train_indices, seq_len)
            val_ds = RandomTextDataset(val_indices, seq_len)

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=bs, shuffle=True, drop_last=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=bs, shuffle=False, drop_last=True
            )

            # Init model
            model = LSTM2Model(vocab_size, emb_dim, h_dim, dropout, device=DEVICE)
            print(f"Model params: {sum(p.numel() for p in model.parameters())}")

            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Sample dir
            sample_dir = os.path.join(grid_dir, "samples", model_name)

            # Train
            training_start = time.time()

            try:
                _, val_losses, val_iters = train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                    n_epochs,
                    model_path,
                    plot_data_path,
                    validate_every=1000,  
                    generate_every=5000,  
                    char_to_idx=char_to_idx,
                    idx_to_char=idx_to_char,
                    raw_text=raw_text,
                    sample_dir=sample_dir,
                )

                training_time = time.time() - training_start

                # Find best validation point
                best_idx = np.argmin(val_losses)
                model_best_val_loss = val_losses[best_idx]
                best_iter = val_iters[best_idx]

                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(
                    val_iters,
                    val_losses,
                    "ro-",
                    label=f"Val Loss (best: {model_best_val_loss:.4f})",
                )
                plt.title(f"LSTM2 lr={lr}, bs={bs}")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.savefig(plot_path, dpi=300)
                plt.close()

                # Generate sample
                sample_file = os.path.join(grid_dir, f"{model_name}_sample.txt")
                text = ""
                try:
                    from utils.generation_utils import generate_text

                    text = generate_text(
                        model,
                        raw_text[:100],
                        idx_to_char,
                        char_to_idx,
                        100,
                        num_chars=1000,
                        temperature=1.0,
                    )
                    with open(sample_file, "w", encoding="utf-8") as f:
                        f.write(text)
                except Exception as e:
                    print(f"Error generating sample: {e}")

                # Save results
                result = {
                    "lr": lr,
                    "bs": bs,
                    "val_loss": model_best_val_loss,
                    "best_iter": best_iter,
                    "time": training_time,
                    "model_path": model_path,
                }
                results.append(result)

                # Update results file
                with open(results_file, "a") as f:
                    f.write(
                        f"{lr:<12} | {bs:<10} | {model_best_val_loss:<8.4f} | {training_time:<15.2f}\n"
                    )

                # Check if best
                if model_best_val_loss < best_val_loss:
                    best_val_loss = model_best_val_loss
                    best_params = (lr, bs)
                    best_model_path = model_path

                # Also save to main model directory
                try:
                    checkpoint = torch.load(model_path, map_location=DEVICE)
                    torch.save(checkpoint, main_model_path)
                    print(f"Model also saved to main directory: {main_model_path}")
                except Exception as e:
                    print(f"Error saving to main directory: {e}")

                print(
                    f"Done in {training_time:.2f}s. Best val loss: {model_best_val_loss:.4f}"
                )

            except Exception as e:
                print(f"Error during training: {e}")
                print(traceback.format_exc())

                # Log error
                with open(os.path.join(grid_dir, "errors.log"), "a") as f:
                    f.write(
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error with lr={lr}, bs={bs}:\n"
                    )
                    f.write(traceback.format_exc())
                    f.write("\n" + "=" * 50 + "\n")

    # Grid search done
    grid_time = time.time() - grid_start
    hours, remainder = divmod(grid_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'-'*50}")
    print(f"Grid search done in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best params: lr={best_params[0]}, bs={best_params[1]}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"{'-'*50}")

    # Save results as DataFrame
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(grid_dir, "grid_search_results.csv"), index=False
        )

        # Create heatmap visualization of results based on validation loss
        plt.figure(figsize=(10, 8))
        heatmap_data = results_df.pivot_table(
            values="val_loss", index="lr", columns="bs"
        )
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu_r", fmt=".4f")
        plt.title("Validation Loss by Learning Rate and Batch Size")
        plt.savefig(os.path.join(grid_dir, "grid_search_loss_heatmap.png"), dpi=300)

        # Create bar plot of all combinations
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=[f"lr={r['lr']}, bs={r['bs']}" for r in results],
            y=[r["val_loss"] for r in results],
        )
        plt.title("Validation Loss by Parameter Combination")
        plt.ylabel("Loss (lower is better)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(grid_dir, "grid_search_loss_barplot.png"), dpi=300)

    # Final summary
    with open(os.path.join(grid_dir, "grid_search_summary.txt"), "w") as f:
        f.write("Grid Search Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(
            f"Best params: lr={best_params[0] if best_params else 'N/A'}, bs={best_params[1] if best_params else 'N/A'}\n"
        )
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n")
        f.write("All Results (sorted by validation loss):\n")
        f.write("-" * 50 + "\n")

        # Sort results by loss
        if len(results) > 0:
            sorted_results = sorted(results, key=lambda x: x["val_loss"])
            for i, r in enumerate(sorted_results):
                f.write(
                    f"{i+1}. lr={r['lr']}, bs={r['bs']} : loss={r['val_loss']:.4f}, time={r['time']:.1f}s\n"
                )

    return best_params, best_model_path
