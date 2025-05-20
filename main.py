import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import pickle
import traceback
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from config import *
from utils.training_utils import train_model, plot_loss, validate
from utils.training_utils import calculate_test_accuracy
from utils.generation_utils import generate_text
from utils.data_utils import load_and_process_data, get_raw_text
from text_analysis import analyze_text_quality
from grid_search import run_grid_search

# Model imports
from Models.rnn_model import RNNModel
from Models.lstm_1_layer import LSTM1Model
from Models.lstm_2_layer import LSTM2Model


def compare_models_by_hidden_dim(models_data, results_dir, model_type):
    print(f"\nComparing {model_type.upper()} models with different hidden dims")

    # Extract plotting data
    dims = [m["hidden_dim"] for m in models_data]
    accs = [m["test_accuracy"] for m in models_data]
    train_losses = [m["train_loss"] for m in models_data]
    val_losses = [m["val_loss"] for m in models_data]
    test_losses = [m["test_loss"] for m in models_data]
    times = [m["training_time"] for m in models_data]

    # Make comparison plots
    plt.figure(figsize=(15, 10))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(dims, accs, marker="o", linestyle="-")
    plt.title(f"{model_type.upper()} Accuracy by Hidden Dimension")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(dims, train_losses, marker="o", linestyle="-", label="Train Loss")
    plt.plot(dims, val_losses, marker="o", linestyle="-", label="Val Loss")
    plt.plot(dims, test_losses, marker="o", linestyle="-", label="Test Loss")
    plt.title(f"{model_type.upper()} Loss by Hidden Dimension")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Training time plot
    plt.subplot(2, 2, 3)
    plt.plot(dims, times, marker="o", linestyle="-")
    plt.title(f"{model_type.upper()} Training Time by Hidden Dimension")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Training Time (s)")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"{model_type}_hidden_dim_comparison.png"), dpi=300
    )
    plt.close()

    # Text summary
    with open(
        os.path.join(results_dir, f"{model_type}_hidden_dim_summary.txt"), "w"
    ) as f:
        f.write(
            f"Comparison of {model_type.upper()} models with different hidden dimensions\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write(
            f"{'Hidden Dim':<15} | {'Accuracy (%)':<15} | {'Train Loss':<15} | {'Val Loss':<15} | {'Test Loss':<15} | {'Training Time (s)':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for i in range(len(dims)):
            f.write(
                f"{dims[i]:<15} | {accs[i]:<15.2f} | {train_losses[i]:<15.4f} | {val_losses[i]:<15.4f} | {test_losses[i]:<15.4f} | {times[i]:<15.2f}\n"
            )


def compare_best_models(model_types, hidden_dims, results_dir):
    print("\nFinding and comparing best models of each type")

    best_models = []

    # Find the best model for each type based on validation loss
    for m_type in model_types:
        best_model = None
        best_val_loss = float("inf")

        for h_dim in hidden_dims:
            model_file = f"{m_type}_hidden{h_dim}_model.pt"
            model_path = os.path.join(MODEL_SAVE_DIR, model_file)

            if os.path.exists(model_path):
                # Try to load the checkpoint
                try:
                    chkpt = torch.load(
                        model_path, map_location=DEVICE, weights_only=False
                    )
                    val_loss = chkpt.get("val_loss", float("inf"))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = {
                            "model_type": m_type,
                            "hidden_dim": h_dim,
                            "val_loss": val_loss,
                            "train_loss": chkpt.get("train_loss", float("inf")),
                            "training_time": chkpt.get("training_time", 0),
                        }
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue

        if best_model:
            best_models.append(best_model)

    if len(best_models) == 0:
        print("No models found for comparison")
        return

    # Create comparison plots
    plt.figure(figsize=(12, 6))

    # Prep the data
    model_names = [
        f"{m['model_type'].upper()} (h={m['hidden_dim']})" for m in best_models
    ]
    val_losses = [m["val_loss"] for m in best_models]
    times = [m["training_time"] for m in best_models]

    # Validation loss bar chart
    plt.subplot(1, 2, 1)
    plt.bar(model_names, val_losses)
    plt.title("Best Model Validation Loss Comparison")
    plt.ylabel("Validation Loss (lower is better)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Training time bar chart
    plt.subplot(1, 2, 2)
    plt.bar(model_names, times)
    plt.title("Best Model Training Time Comparison")
    plt.ylabel("Training Time (s)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "best_models_comparison.png"), dpi=300)
    plt.close()

    # Text summary too
    with open(os.path.join(results_dir, "best_models_summary.txt"), "w") as f:
        f.write("Comparison of Best Models of Each Type\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"{'Model Type':<15} | {'Hidden Dim':<15} | {'Val Loss':<15} | {'Train Loss':<15} | {'Training Time (s)':<15}\n"
        )
        f.write("-" * 100 + "\n")

        for m in best_models:
            f.write(
                f"{m['model_type']:<15} | {m['hidden_dim']:<15} | {m['val_loss']:<15.4f} | {m['train_loss']:<15.4f} | {m['training_time']:<15.2f}\n"
            )


def run_experiment(test_mode=False):
    # Start timer
    start = time.time()

    # Model config
    model_types = ["lstm2"] 
    hidden_dims = [65, 128, 256] if not test_mode else [64]

    # Create dirs for text samples
    samples_training_dir = os.path.join(SAMPLE_SAVE_DIR, "training_samples")
    samples_temp_dir = os.path.join(SAMPLE_SAVE_DIR, "temperature_comparison")
    samples_nucleus_dir = os.path.join(SAMPLE_SAVE_DIR, "nucleus_comparison")
    os.makedirs(samples_training_dir, exist_ok=True)
    os.makedirs(samples_temp_dir, exist_ok=True)
    os.makedirs(samples_nucleus_dir, exist_ok=True)

    # Setup experiment results dir
    results_dir = os.path.join(MODEL_SAVE_DIR, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    # Init results file
    results_file = os.path.join(results_dir, "results_summary.txt")
    with open(results_file, "w") as f:
        f.write(
            "Model Type | Hidden Dim | Test Loss | Best Val Loss | Train Loss | Test Accuracy (%) | Training Time (s)\n"
        )
        f.write("-" * 100 + "\n")

    # Load data
    print("Loading and preprocessing data")
    loaders, char_to_idx, idx_to_char, vocab_size = load_and_process_data()
    train_loader, val_loader, test_loader = loaders

    # Update global vocab size
    global VOCAB_SIZE
    VOCAB_SIZE = vocab_size
    print(f"Vocab size: {VOCAB_SIZE}")

    # Load text for generation
    raw_text = get_raw_text()

    # Set training params based on test mode
    num_epochs = 2 if test_mode else NUM_EPOCHS

    # Train each model type
    for m_type in model_types:
        print(f"\n{'='*50}")
        print(f"TESTING MODEL: {m_type.upper()}")
        print(f"{'='*50}")

        # Store results for hidden dim comparison
        model_results = []

        # Train for each hidden dim
        for h_dim in hidden_dims:
            try:
                print(f"\n{'-'*50}")
                print(f"Training {m_type.upper()} with hidden_dim={h_dim}")
                print(f"{'-'*50}")

                # Setup file paths
                model_filename = f"{m_type}_hidden{h_dim}_model.pt"
                plot_filename = f"{m_type}_hidden{h_dim}_metrics.png"
                plot_data_filename = f"{m_type}_hidden{h_dim}_plot_data.pkl"

                model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
                plot_path = os.path.join(PLOT_SAVE_DIR, plot_filename)
                plot_data_path = os.path.join(PLOT_SAVE_DIR, "data", plot_data_filename)

                # Sample dir for this model
                model_sample_dir = os.path.join(
                    samples_training_dir, f"{m_type}_hidden{h_dim}"
                )

                # Init the right model type
                if m_type == "rnn":
                    model = RNNModel(
                        VOCAB_SIZE, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                    )
                elif m_type == "lstm1":
                    model = LSTM1Model(
                        VOCAB_SIZE, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                    )
                elif m_type == "lstm2":
                    model = LSTM2Model(
                        VOCAB_SIZE, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                    )

                n_params = sum(p.numel() for p in model.parameters())
                print(
                    f"Model: {m_type.upper()} (hidden_dim={h_dim}) with {n_params} parameters"
                )
                print(f"Using device: {DEVICE}")

                # Setup training
                criterion = nn.CrossEntropyLoss()
                optim_params = {"lr": LEARNING_RATE}
                optimizer = optim.Adam(model.parameters(), **optim_params)

                # Check if model already exists
                if os.path.exists(model_path) and not test_mode:
                    print(f"Found existing model at {model_path}, loading")
                    try:
                        chkpt = torch.load(
                            model_path, map_location=DEVICE, weights_only=False
                        )
                        model.load_state_dict(chkpt["model_state_dict"])

                        # Load plot data if exists
                        if os.path.exists(plot_data_path):
                            with open(plot_data_path, "rb") as f:
                                plot_data = pickle.load(f)

                            # Plot metrics
                            plot_loss(
                                plot_data.get("train_losses", []),
                                plot_data.get("val_losses", []),
                                plot_data.get("val_iterations", []),
                                title=f"{m_type.upper()} (hidden_dim={h_dim})",
                                save_path=plot_path,
                            )

                        # Get metrics from checkpoint
                        training_time = chkpt.get("training_time", 0)
                        val_loss = chkpt.get("val_loss", float("inf"))
                        train_loss = chkpt.get("train_loss", float("inf"))

                    except Exception as e:
                        print(f"Error loading model: {e}")
                        print("Will train a new model instead.")
                        training_time = 0
                        val_loss = float("inf")
                        train_loss = float("inf")
                else:
                    # Train the model
                    print(f"Training for {num_epochs} epochs")
                    train_start = time.time()

                    train_losses, val_losses, val_iterations = train_model(
                        model,
                        train_loader,
                        val_loader,
                        criterion,
                        optimizer,
                        DEVICE,
                        num_epochs,
                        model_path,
                        plot_data_path,
                        validate_every=1000,
                        generate_every=10000,
                        char_to_idx=char_to_idx,
                        idx_to_char=idx_to_char,
                        raw_text=raw_text,
                        sample_dir=model_sample_dir,
                    )

                    training_time = time.time() - train_start

                    # Calculate final losses
                    train_loss = (
                        np.mean(train_losses[-100:]) if train_losses else float("inf")
                    )
                    val_loss = val_losses[-1] if val_losses else float("inf")

                    # Update checkpoint with training time
                    try:
                        chkpt = torch.load(
                            model_path, map_location=DEVICE, weights_only=False
                        )
                        chkpt["training_time"] = training_time
                        chkpt["train_loss"] = train_loss
                        chkpt["val_loss"] = val_loss
                        torch.save(chkpt, model_path)
                    except Exception as e:
                        print(
                            f"Warning: Couldn't update checkpoint with training time: {e}"
                        )

                    # Plot metrics
                    plot_loss(
                        train_losses,
                        val_losses,
                        val_iterations,
                        title=f"{m_type.upper()} (hidden_dim={h_dim})",
                        save_path=plot_path,
                    )

                # Evaluate on test set
                print("Evaluating on test set")
                test_loss = validate(model, test_loader, criterion, DEVICE)
                test_acc = calculate_test_accuracy(model, test_loader, DEVICE)

                print(f"Test Results: Loss={test_loss:.4f}, Accuracy={test_acc:.2f}%")

                # Log results
                with open(results_file, "a") as f:
                    f.write(
                        f"{m_type:<10} | {h_dim:<10} | {test_loss:<15.4f} | {val_loss:<12.4f} | {train_loss:<10.4f} | {test_acc:<18.2f} | {training_time:<15.2f}\n"
                    )

                # Store model data for comparison
                model_results.append(
                    {
                        "model_type": m_type,
                        "hidden_dim": h_dim,
                        "model": model,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "training_time": training_time,
                    }
                )

                # Generate text samples for the largest model of each type
                if h_dim == max(hidden_dims) or test_mode:
                    print("Generating text samples with different parameters")

                    # Setup sample dirs
                    temp_sample_dir = os.path.join(samples_temp_dir, m_type)
                    nucleus_sample_dir = os.path.join(samples_nucleus_dir, m_type)
                    os.makedirs(temp_sample_dir, exist_ok=True)
                    os.makedirs(nucleus_sample_dir, exist_ok=True)

                    # Generate with different temps
                    temps = [0.5, 0.7, 1.0, 1.3, 1.5]
                    for temp in temps:
                        sample_file = os.path.join(
                            temp_sample_dir, f"temp_{temp:.1f}.txt"
                        )
                        print(f"Generating with temp={temp}")
                        text = generate_text(
                            model,
                            raw_text[:100],
                            idx_to_char,
                            char_to_idx,
                            100,
                            num_chars=1000,
                            temp=temp,
                            use_nucleus=False,
                        )
                        with open(sample_file, "w", encoding="utf-8") as f:
                            f.write(text)

                    # Generate with different nucleus params
                    p_values = [0.5, 0.7, 0.9, 0.95]
                    for p in p_values:
                        sample_file = os.path.join(
                            nucleus_sample_dir, f"nucleus_{p:.2f}.txt"
                        )
                        print(f"Generating with nucleus p={p}")
                        text = generate_text(
                            model,
                            raw_text[:100],
                            idx_to_char,
                            char_to_idx,
                            100,
                            num_chars=1000,
                            temp=1.0,
                            use_nucleus=True,
                            p=p,
                        )
                        with open(sample_file, "w", encoding="utf-8") as f:
                            f.write(text)

                    # Analyze text quality
                    print("Analyzing text quality metrics")
                    quality_metrics = {}

                    # Temp metrics
                    for temp in temps:
                        sample_file = os.path.join(
                            temp_sample_dir, f"temp_{temp:.1f}.txt"
                        )
                        with open(sample_file, "r", encoding="utf-8") as f:
                            gen_text = f.read()

                        metrics = analyze_text_quality(gen_text, raw_text)
                        quality_metrics[f"temperature_{temp:.1f}"] = metrics

                    # Nucleus metrics
                    for p in p_values:
                        sample_file = os.path.join(
                            nucleus_sample_dir, f"nucleus_{p:.2f}.txt"
                        )
                        with open(sample_file, "r", encoding="utf-8") as f:
                            gen_text = f.read()

                        metrics = analyze_text_quality(gen_text, raw_text)
                        quality_metrics[f"nucleus_{p:.2f}"] = metrics

                    # Save quality metrics
                    quality_file = os.path.join(
                        results_dir, f"{m_type}_quality_metrics.pkl"
                    )
                    with open(quality_file, "wb") as f:
                        pickle.dump(quality_metrics, f)

                    # Create summary
                    summary_file = os.path.join(
                        results_dir, f"{m_type}_quality_summary.txt"
                    )
                    with open(summary_file, "w") as f:
                        f.write(
                            f"Quality Metrics for {m_type.upper()} (hidden_dim={h_dim})\n"
                        )
                        f.write("=" * 80 + "\n\n")

                        f.write("Temperature Sampling:\n")
                        f.write("-" * 50 + "\n")
                        f.write(
                            f"{'Temp':<10} | {'Char Div':<10} | {'Word Div':<10} | {'Bigram Rep':<12} | {'Trigram Rep':<12} | {'Bigram Overlap':<15}\n"
                        )
                        for temp in temps:
                            metrics = quality_metrics[f"temperature_{temp:.1f}"]
                            f.write(
                                f"{temp:<10.1f} | {metrics['char_diversity']:<10.4f} | {metrics['word_diversity']:<10.4f} | {metrics['bigram_repetition']:<12.4f} | {metrics['trigram_repetition']:<12.4f} | {metrics.get('bigram_overlap', 0.0):<15.4f}\n"
                            )

                        f.write("\nNucleus Sampling:\n")
                        f.write("-" * 50 + "\n")
                        f.write(
                            f"{'p-value':<10} | {'Char Div':<10} | {'Word Div':<10} | {'Bigram Rep':<12} | {'Trigram Rep':<12} | {'Bigram Overlap':<15}\n"
                        )
                        for p in p_values:
                            metrics = quality_metrics[f"nucleus_{p:.2f}"]
                            f.write(
                                f"{p:<10.2f} | {metrics['char_diversity']:<10.4f} | {metrics['word_diversity']:<10.4f} | {metrics['bigram_repetition']:<12.4f} | {metrics['trigram_repetition']:<12.4f} | {metrics.get('bigram_overlap', 0.0):<15.4f}\n"
                            )

            except Exception as e:
                print(f"\nERROR training {m_type} with hidden_dim={h_dim}:")
                print(traceback.format_exc())

                # Log the error
                with open(os.path.join(results_dir, "errors.log"), "a") as f:
                    f.write(
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error training {m_type} (hidden_dim={h_dim}):\n"
                    )
                    f.write(traceback.format_exc())
                    f.write("\n" + "=" * 50 + "\n")

                continue

        # Compare all models of this type
        if len(model_results) > 1:
            compare_models_by_hidden_dim(model_results, results_dir, m_type)

    # Do a final comparison of best models of each type
    compare_best_models(model_types, hidden_dims, results_dir)

    # Report total runtime
    total_time = time.time() - start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*50}")
    print(f"Experiment finished in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to {results_file}")
    print(f"{'='*50}")

    return True


def generate_samples_for_all_models(sample_length=1000, seed_len=100):
    """Generate samples for all models in MODEL_SAVE_DIR"""
    print(f"\n{'='*50}")
    print("GENERATING SAMPLES FOR ALL SAVED MODELS")
    print(f"{'='*50}")

    # Load data for chars
    _, char_to_idx, idx_to_char, vocab_size = load_and_process_data()

    # Get raw text for seed
    raw_text = get_raw_text()
    if not raw_text:
        print("Can't load raw text for seed. Aborting.")
        return

    # Temps and p-values to try
    temps = [0.5, 0.7, 1.0, 1.3, 1.5]
    p_vals = [0.5, 0.7, 0.9, 0.95]

    # Setup dirs
    os.makedirs(os.path.join(SAMPLE_SAVE_DIR, "temperature_comparison"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_SAVE_DIR, "nucleus_comparison"), exist_ok=True)

    # Find model files
    model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith(".pt")]

    # For each model
    for m_file in model_files:
        try:
            # Parse model type & hidden dim from filename
            m_parts = m_file.split("_")
            if len(m_parts) < 3 or not m_parts[1].startswith("hidden"):
                print(f"Skipping weird filename: {m_file}")
                continue

            m_type = m_parts[0]
            h_dim = int(m_parts[1].replace("hidden", ""))

            print(f"\n{'-'*50}")
            print(f"Generating for {m_type.upper()} with hidden_dim={h_dim}")
            print(f"{'-'*50}")

            # Setup dirs for this model
            temp_dir = os.path.join(SAMPLE_SAVE_DIR, "temperature_comparison", m_type)
            nucleus_dir = os.path.join(SAMPLE_SAVE_DIR, "nucleus_comparison", m_type)
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(nucleus_dir, exist_ok=True)

            # Load the model
            m_path = os.path.join(MODEL_SAVE_DIR, m_file)
            print(f"Loading from {m_path}")

            # Init right model type
            if m_type == "rnn":
                model = RNNModel(
                    vocab_size, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                )
            elif m_type == "lstm1":
                model = LSTM1Model(
                    vocab_size, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                )
            elif m_type == "lstm2" or m_type == "lstm2best": 
                model = LSTM2Model(
                    vocab_size, EMBEDDING_DIM, h_dim, DROPOUT, device=DEVICE
                )

            # Load weights
            try:
                chkpt = torch.load(m_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(chkpt["model_state_dict"])
                model.eval()
            except Exception as e:
                print(f"Failed to load model: {e}")
                continue

            # Generate with different temps
            print("Generating temperature samples")
            for temp in temps:
                out_file = os.path.join(temp_dir, f"temp_{temp:.1f}.txt")
                text = generate_text(
                    model,
                    raw_text[:seed_len],
                    idx_to_char,
                    char_to_idx,
                    seed_len,
                    num_chars=sample_length,
                    temp=temp,
                    use_nucleus=False,
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"temp={temp:.1f}")

            # Generate with nucleus sampling
            print("Generating nucleus samples")
            for p in p_vals:
                out_file = os.path.join(nucleus_dir, f"nucleus_{p:.2f}.txt")
                text = generate_text(
                    model,
                    raw_text[:seed_len],
                    idx_to_char,
                    char_to_idx,
                    seed_len,
                    num_chars=sample_length,
                    temp=1.0,
                    use_nucleus=True,
                    p=p,
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"nucleus p={p:.2f}")

            # Analyze quality
            quality_metrics = {}

            # Temp metrics
            for temp in temps:
                with open(
                    os.path.join(temp_dir, f"temp_{temp:.1f}.txt"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    gen_text = f.read()
                metrics = analyze_text_quality(gen_text, raw_text)
                quality_metrics[f"temperature_{temp:.1f}"] = metrics

            # Nucleus metrics
            for p in p_vals:
                with open(
                    os.path.join(nucleus_dir, f"nucleus_{p:.2f}.txt"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    gen_text = f.read()
                metrics = analyze_text_quality(gen_text, raw_text)
                quality_metrics[f"nucleus_{p:.2f}"] = metrics

            # Save quality data
            quality_dir = os.path.join(MODEL_SAVE_DIR, "quality_metrics")
            os.makedirs(quality_dir, exist_ok=True)

            quality_file = os.path.join(
                quality_dir, f"{m_type}_hidden{h_dim}_quality.pkl"
            )
            with open(quality_file, "wb") as f:
                pickle.dump(quality_metrics, f)

            # Generate summary text file
            summary_file = os.path.join(
                quality_dir, f"{m_type}_hidden{h_dim}_quality.txt"
            )
            with open(summary_file, "w") as f:
                f.write(f"Quality Metrics for {m_type.upper()} (hidden_dim={h_dim})\n")
                f.write("=" * 80 + "\n\n")

                f.write("Temperature Sampling:\n")
                f.write("-" * 50 + "\n")
                f.write(
                    f"{'Temp':<10} | {'Char Div':<10} | {'Word Div':<10} | {'Bigram Rep':<12} | {'Trigram Rep':<12} | {'Bigram Overlap':<15}\n"
                )
                for temp in temps:
                    m = quality_metrics[f"temperature_{temp:.1f}"]
                    f.write(
                        f"{temp:<10.1f} | {m['char_diversity']:<10.4f} | {m['word_diversity']:<10.4f} | {m['bigram_repetition']:<12.4f} | {m['trigram_repetition']:<12.4f} | {m.get('bigram_overlap', 0.0):<15.4f}\n"
                    )

                f.write("\nNucleus Sampling:\n")
                f.write("-" * 50 + "\n")
                f.write(
                    f"{'p-value':<10} | {'Char Div':<10} | {'Word Div':<10} | {'Bigram Rep':<12} | {'Trigram Rep':<12} | {'Bigram Overlap':<15}\n"
                )
                for p in p_vals:
                    m = quality_metrics[f"nucleus_{p:.2f}"]
                    f.write(
                        f"{p:<10.2f} | {m['char_diversity']:<10.4f} | {m['word_diversity']:<10.4f} | {m['bigram_repetition']:<12.4f} | {m['trigram_repetition']:<12.4f} | {m.get('bigram_overlap', 0.0):<15.4f}\n"
                    )

            print(f"Quality data saved to {quality_file}")

        except Exception as e:
            print(f"Error for {m_file}: {str(e)}")
            print(traceback.format_exc())

    print(f"\n{'='*50}")
    print("ALL SAMPLES GENERATED")
    print(f"{'='*50}")


def main():
    TEST_MODE = True
    RUN_GRID_SEARCH = False
    GENERATE_SAMPLES = False

    if RUN_GRID_SEARCH:
        print("Running grid search")
        best_params, best_model_path = run_grid_search()
        print(f"\nGrid search complete")
        print(f"Best params: lr={best_params[0]}, batch_size={best_params[1]}")
        print(f"Best model saved at: {best_model_path}")
    elif GENERATE_SAMPLES:
        generate_samples_for_all_models()
    else:
        print("Starting main experiment")
        run_experiment(test_mode=TEST_MODE)


if __name__ == "__main__":
    main()
