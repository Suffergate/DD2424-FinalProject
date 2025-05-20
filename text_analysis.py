import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
from nltk.util import ngrams
from tqdm import tqdm
from config import *
import enchant

# Try to get NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def load_ref_text():
    """Load reference text for comparison"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Can't load reference text: {e}")
        return None


def get_sample_files():
    """Find all the generated sample files"""
    samples = []

    # Temperature samples
    temp_dir = os.path.join(SAMPLE_SAVE_DIR, "temperature_comparison")
    for model in os.listdir(temp_dir):
        model_dir = os.path.join(temp_dir, model)
        if os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".txt"):
                    samples.append(
                        {
                            "path": os.path.join(model_dir, file),
                            "model": model,
                            "method": "temperature",
                            "param": float(
                                file.replace("temp_", "").replace(".txt", "")
                            ),
                        }
                    )

    # Nucleus samples
    nuc_dir = os.path.join(SAMPLE_SAVE_DIR, "nucleus_comparison")
    for model in os.listdir(nuc_dir):
        model_dir = os.path.join(nuc_dir, model)
        if os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".txt"):
                    samples.append(
                        {
                            "path": os.path.join(model_dir, file),
                            "model": model,
                            "method": "nucleus",
                            "param": float(
                                file.replace("nucleus_", "").replace(".txt", "")
                            ),
                        }
                    )

    return samples


def check_spelling(text):
    """Check spelling with enchant"""
    dict = enchant.Dict("en_US")

    # Get words
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0, 0, []

    # Check each word
    good = []
    bad = []
    for word in words:
        if dict.check(word):
            good.append(word)
        else:
            bad.append(word)

    # Get percentage
    pct = len(good) / len(words) * 100

    return pct, len(words), bad


def calc_ngram_coverage(gen_text, ref_text, n=2):
    """How many of the generated n-grams appear in the reference text"""
    # Lowercase everything
    gen_text = gen_text.lower()
    ref_text = ref_text.lower()

    # Get n-grams
    gen_grams = list(ngrams(gen_text, n))
    ref_grams = set(ngrams(ref_text, n))

    # Count matches
    matches = sum(1 for g in gen_grams if g in ref_grams)

    if len(gen_grams) == 0:
        return 0

    return matches / len(gen_grams) * 100


def analyze_repetition(text):
    """Check how repetitive the text is"""
    # Character repetition
    char_bigrams = list(ngrams(text.lower(), 2))
    char_trigrams = list(ngrams(text.lower(), 3))

    bigram_rep = 1 - (len(set(char_bigrams)) / len(char_bigrams)) if char_bigrams else 0
    trigram_rep = (
        1 - (len(set(char_trigrams)) / len(char_trigrams)) if char_trigrams else 0
    )

    # Word repetition
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_bigrams = list(ngrams(words, 2)) if len(words) >= 2 else []
    word_trigrams = list(ngrams(words, 3)) if len(words) >= 3 else []

    word_bigram_rep = (
        1 - (len(set(word_bigrams)) / len(word_bigrams)) if word_bigrams else 0
    )
    word_trigram_rep = (
        1 - (len(set(word_trigrams)) / len(word_trigrams)) if word_trigrams else 0
    )

    return {
        "char_bigram_rep": bigram_rep * 100,
        "char_trigram_rep": trigram_rep * 100,
        "word_bigram_rep": word_bigram_rep * 100,
        "word_trigram_rep": word_trigram_rep * 100,
    }


def analyze_all_samples():
    """Analyze all the sample files"""
    print("Starting text quality analysis")

    # Get reference
    ref_text = load_ref_text()
    if not ref_text:
        print("ERROR: Can't load reference text!")
        return

    # Find samples
    samples = get_sample_files()
    print(f"Found {len(samples)} sample files")

    # Results storage
    results = []

    # Loop through samples
    for sample in tqdm(samples, desc="Analyzing"):
        try:
            with open(sample["path"], "r", encoding="utf-8") as f:
                text = f.read()

            # Skip empty files
            if not text.strip():
                continue

            # Remove seed if present
            if "\n\nGenerated text" in text:
                text = text.split("\n\nGenerated text")[1]

            # Basic stats
            char_count = len(text)

            # Spelling check
            spell_pct, word_count, misspelled = check_spelling(text)

            # N-gram coverage
            bigram_cov = calc_ngram_coverage(text, ref_text, 2)
            trigram_cov = calc_ngram_coverage(text, ref_text, 3)

            # Repetition
            rep_metrics = analyze_repetition(text)

            # Store results
            result = {
                "model": sample["model"],
                "method": sample["method"],
                "char_count": char_count,
                "word_count": word_count,
                "correct_spelling_pct": spell_pct,
                "bigram_coverage": bigram_cov,
                "trigram_coverage": trigram_cov,
                **rep_metrics,  # Add all repetition metrics
            }

            # Add parameter 
            if "param" in sample:
                result["param"] = sample["param"]
                results.append(result)

        except Exception as e:
            print(f"Error analyzing {sample['path']}: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(ANALYSIS_SAVE_DIR, "text_quality_results.csv"), index=False
        )

        # Create viz
        create_visualizations(results_df)
    else:
        print("No results to save!")

    return results


def create_visualizations(df):
    """Create plots of the analysis results"""
    print("Making plots")

    # Make plots dir
    plots_dir = os.path.join(ANALYSIS_SAVE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if df.empty:
        print("No data to plot.")
        return

    # Spelling comparison by model and method
    plt.figure(figsize=(12, 8))

    for method in df["method"].unique():
        method_data = df[df["method"] == method]

        # Group by model and param
        pivot = method_data.pivot(
            index="param", columns="model", values="correct_spelling_pct"
        )

        plt.subplot(1, 2, 1 if method == "temperature" else 2)
        pivot.plot(marker="o")
        plt.title(f"Spelling: {method.capitalize()}")
        plt.xlabel(
            "Parameter"
            + (" (Temperature)" if method == "temperature" else " (p-value)")
        )
        plt.ylabel("Correct Words (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Model")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "spelling_comparison.png"), dpi=300)
    plt.close()

    # Overall quality heatmap
    # Normalize metrics
    norm_df = df.copy()

    # Metrics where higher is better
    better_metrics = [
        "correct_spelling_pct",
        "bigram_coverage",
        "trigram_coverage",
    ]

    for metric in better_metrics:
        min_val = norm_df[metric].min()
        max_val = norm_df[metric].max()
        if max_val > min_val:
            norm_df[metric] = (norm_df[metric] - min_val) / (max_val - min_val)

    # Metrics where lower is better
    worse_metrics = [
        "char_bigram_rep",
        "char_trigram_rep",
        "word_bigram_rep",
        "word_trigram_rep",
    ]

    for metric in worse_metrics:
        min_val = norm_df[metric].min()
        max_val = norm_df[metric].max()
        if max_val > min_val:
            norm_df[metric] = 1 - (norm_df[metric] - min_val) / (max_val - min_val)

    # Overall quality score
    all_metrics = better_metrics + worse_metrics
    norm_df["quality"] = norm_df[all_metrics].mean(axis=1)

    # Create heatmap for each method
    for method in norm_df["method"].unique():
        method_data = norm_df[norm_df["method"] == method]

        # Create pivot for heatmap
        hm_data = method_data.pivot(index="model", columns="param", values="quality")

        plt.figure(figsize=(10, 6))
        sns.heatmap(hm_data, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Overall Text Quality: {method.capitalize()}")
        plt.xlabel(
            "Parameter"
            + (" (Temperature)" if method == "temperature" else " (p-value)")
        )
        plt.ylabel("Model Type")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"quality_heatmap_{method}.png"), dpi=300)
        plt.close()


def analyze_text_quality(text, ref_text=None):
    """Get quality metrics for a single text"""
    # Char diversity (% unique chars)
    char_div = len(set(text)) / len(text) if len(text) > 0 else 0

    # Get words
    words = re.findall(r"\b\w+\b", text.lower())
    ref_words = re.findall(r"\b\w+\b", ref_text.lower()) if ref_text else []

    # Word diversity
    word_div = len(set(words)) / len(words) if len(words) > 0 else 0

    # Repetition
    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    bigram_rep = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0

    trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
    trigram_rep = 1 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0

    metrics = {
        "char_diversity": char_div,
        "word_diversity": word_div,
        "bigram_repetition": bigram_rep,
        "trigram_repetition": trigram_rep,
        "length": len(text),
    }

    # Add reference comparison metrics if we have a reference
    if ref_text:
        # Bigram overlap
        text_bigrams = set(bigrams)
        ref_bigrams = set([ref_text[i : i + 2] for i in range(len(ref_text) - 1)])

        if ref_bigrams:
            metrics["bigram_overlap"] = len(
                text_bigrams.intersection(ref_bigrams)
            ) / len(ref_bigrams)
        else:
            metrics["bigram_overlap"] = 0

        # Word overlap
        text_wordset = set(words)
        ref_wordset = set(ref_words)

        if ref_wordset:
            metrics["word_overlap"] = len(text_wordset.intersection(ref_wordset)) / len(
                ref_wordset
            )
        else:
            metrics["word_overlap"] = 0

    return metrics


def main():
    """Main analysis function"""
    results = analyze_all_samples()

    if results and len(results) > 0:
        # Find the best in each category
        df = pd.DataFrame(results)

        print("\nAnalysis Results:")
        print("-" * 50)

        # Best spelling
        best_spell = df.loc[df["correct_spelling_pct"].idxmax()]
        print(
            f"Best spelling: {best_spell['model']} with {best_spell['method']} "
            f"(param={best_spell['param']:.2f}), "
            f"correct words: {best_spell['correct_spelling_pct']:.2f}%"
        )

        # Best bigram coverage
        best_bigram = df.loc[df["bigram_coverage"].idxmax()]
        print(
            f"Best bigram coverage: {best_bigram['model']} with {best_bigram['method']} "
            f"(param={best_bigram['param']:.2f}), "
            f"coverage: {best_bigram['bigram_coverage']:.2f}%"
        )

        # Least repetition
        best_rep = df.loc[df["char_bigram_rep"].idxmin()]
        print(
            f"Least repetitive: {best_rep['model']} with {best_rep['method']} "
            f"(param={best_rep['param']:.2f}), "
            f"char bigram rep: {best_rep['char_bigram_rep']:.2f}%"
        )

        print(f"\nComplete results saved to {ANALYSIS_SAVE_DIR}")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
