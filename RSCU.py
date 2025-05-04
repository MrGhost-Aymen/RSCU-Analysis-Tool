#!/usr/bin/env python3
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import textwrap
import os
from multiprocessing import Pool
from functools import partial
import json
from matplotlib.colors import LinearSegmentedColormap

# Valid amino acid codes (including stop codon)
VALID_AA_CODES = set('ACDEFGHIKLMNPQRSTVWY*')

# Mapping of one-letter to three-letter amino acid codes
AA_1_TO_3 = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
    'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
    'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
    '*': 'End'  # Stop codon
}

# Load genetic codes from JSON file
with open('genetic_codes.json', 'r') as f:
    GENETIC_CODES = json.load(f)

def validate_genetic_codes():
    """Validate all genetic code tables for correct amino acid codes."""
    for code_name, code in GENETIC_CODES.items():
        invalid_aas = [aa for aa in code['table'].values() if aa not in VALID_AA_CODES]
        if invalid_aas:
            raise ValueError(
                f"Invalid amino acid codes in {code_name}: {', '.join(invalid_aas)}. "
                f"Valid codes: {', '.join(sorted(VALID_AA_CODES))}"
            )
        if not code['start']:
            raise ValueError(f"No start codons defined for {code_name}")

# Run validation at script startup
validate_genetic_codes()

def get_genetic_code(name='standard'):
    """Return the specified genetic code dictionary."""
    if name not in GENETIC_CODES:
        raise ValueError(f"Invalid genetic code: {name}. Available codes: {', '.join(GENETIC_CODES.keys())}")
    return GENETIC_CODES.get(name, GENETIC_CODES['standard'])

def reverse_table(genetic_code):
    """Create amino acid to codons mapping, excluding stop codons."""
    aa_to_codons = defaultdict(list)
    for codon, aa in genetic_code['table'].items():
        if aa != '*':
            aa_to_codons[aa].append(codon)
    return aa_to_codons

def calculate_gc_content(seq):
    """Calculate GC content of a sequence."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    total = len(seq)
    return (gc_count / total * 100) if total > 0 else 0

def process_sequence(record, genetic_code, min_length):
    """Process a single sequence for codon counting and GC content."""
    codon_count = defaultdict(int)
    invalid_codons = set()
    seq = str(record.seq).upper()
    if 'U' in seq:  # Convert RNA to DNA
        seq = seq.replace('U', 'T')
    
    # Check sequence length
    if len(seq) < min_length:
        return None, None, 0, set()
    
    if len(seq) % 3 != 0:
        print(f"Warning: Sequence {record.id} length {len(seq)} is not a multiple of 3")
        return None, None, 0, set()
    
    seq_length = len(seq) // 3
    gc_content = calculate_gc_content(seq)
    
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in genetic_code['table']:
            if genetic_code['table'][codon] != '*':
                codon_count[codon] += 1
        else:
            invalid_codons.add(codon)
    
    return codon_count, gc_content, seq_length, invalid_codons

def parse_sequences(fasta_file, genetic_code_name='standard', min_length=30, parallel=False):
    """Parse sequences and count codons, optionally in parallel."""
    genetic_code = get_genetic_code(genetic_code_name)
    codon_count = defaultdict(int)
    total_codons = 0
    seq_lengths = []
    gc_contents = []
    invalid_codons = set()
    
    try:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {fasta_file}")
    
    if not sequences:
        raise ValueError("No sequences found in input file")
    
    if parallel:
        with Pool() as pool:
            results = pool.starmap(
                partial(process_sequence, genetic_code=genetic_code, min_length=min_length),
                [(record, genetic_code, min_length) for record in sequences]
            )
    else:
        results = [process_sequence(record, genetic_code, min_length) for record in sequences]
    
    for result in results:
        if result[0] is None:  # Skipped sequence
            continue
        seq_codon_count, gc_content, seq_length, seq_invalid_codons = result
        for codon, count in seq_codon_count.items():
            codon_count[codon] += count
            total_codons += count
        seq_lengths.append(seq_length)
        gc_contents.append(gc_content)
        invalid_codons.update(seq_invalid_codons)
    
    if invalid_codons:
        print(f"Warning: Found {len(invalid_codons)} invalid codons: {', '.join(sorted(invalid_codons))}")
    
    return codon_count, total_codons, seq_lengths, gc_contents

def calculate_rscu(codon_count, genetic_code_name='standard'):
    """Calculate Relative Synonymous Codon Usage (RSCU)."""
    genetic_code = get_genetic_code(genetic_code_name)
    aa_to_codons = reverse_table(genetic_code)
    rscu = {}
    
    for aa, codons in aa_to_codons.items():
        total = sum(codon_count.get(codon, 0) for codon in codons)
        n = len(codons)
        
        if total == 0:
            for codon in codons:
                rscu[codon] = 0.0
        else:
            for codon in codons:
                observed = codon_count.get(codon, 0)
                expected = total / n
                rscu[codon] = min(observed / expected if expected != 0 else 0.0, 2.5)  # Cap RSCU at 2.5
    
    return rscu

def calculate_cai(codon_count, reference_weights, genetic_code_name='standard'):
    """Calculate Codon Adaptation Index (CAI) using reference weights."""
    genetic_code = get_genetic_code(genetic_code_name)
    
    if not reference_weights:
        print("Warning: No reference weights provided for CAI calculation")
        return None
    
    cai = 0
    total = 0
    
    for codon, count in codon_count.items():
        aa = genetic_code['table'].get(codon, 'X')
        if aa != '*' and codon in reference_weights:
            weight = reference_weights.get(codon, 1.0)
            if weight > 0:  # Avoid log(0)
                cai += count * np.log(weight)
                total += count
    
    return np.exp(cai / total) if total > 0 else None

def calculate_enc(codon_count, genetic_code_name='standard'):
    """Calculate Effective Number of Codons (ENc)."""
    genetic_code = get_genetic_code(genetic_code_name)
    aa_to_codons = reverse_table(genetic_code)
    f_values = []
    
    for aa, codons in aa_to_codons.items():
        n = len(codons)
        if n == 1:
            continue
            
        total = sum(codon_count.get(codon, 0) for codon in codons)
        if total == 0:
            continue
            
        s = sum((codon_count.get(codon, 0) / total)**2 for codon in codons)
        f = (s * n - 1) / (n - 1) if n > 1 else 1.0
        f_values.append(f)
    
    if not f_values:
        return None
        
    F = sum(f_values) / len(f_values)
    enc = 2 + 9/F if F != 0 else 61.0
    return min(enc, 61.0)

def generate_stats(codon_count, total_codons, seq_lengths, gc_contents, cai, genetic_code_name='standard'):
    """Generate comprehensive statistics including GC content."""
    genetic_code = get_genetic_code(genetic_code_name)
    aa_to_codons = reverse_table(genetic_code)
    
    stats = {
        'total_sequences': len(seq_lengths),
        'total_codons': total_codons,
        'avg_seq_length': np.mean(seq_lengths) if seq_lengths else 0,
        'median_seq_length': np.median(seq_lengths) if seq_lengths else 0,
        'avg_gc_content': np.mean(gc_contents) if gc_contents else 0,
        'enc': calculate_enc(codon_count, genetic_code_name),
        'cai': cai
    }
    
    codon_percent = {codon: (count / total_codons * 100) if total_codons else 0
                    for codon, count in codon_count.items()}
    
    aa_usage = defaultdict(int)
    for codon, count in codon_count.items():
        aa = genetic_code['table'].get(codon, 'X')
        aa_usage[aa] += count
    
    return stats, codon_percent, aa_usage

def plot_rscu(rscu, genetic_code_name='standard', output_prefix='rscu', plot_type='all'):
    """Create multiple visualizations of RSCU values."""
    genetic_code = get_genetic_code(genetic_code_name)

    # Prepare data for plotting
    data = []
    for codon, value in rscu.items():
        aa = genetic_code['table'].get(codon, 'X')
        data.append({
            'Codon': codon,
            'Amino Acid': aa,
            'RSCU': value,
            'Type': 'Stop' if aa == '*' else 'Normal'
        })
    
    df = pd.DataFrame(data)
    df = df[df['Type'] == 'Normal']
    df['Amino Acid 3'] = df['Amino Acid'].map(AA_1_TO_3)
    df.sort_values(['Amino Acid', 'RSCU'], ascending=[True, False], inplace=True)

    # Set up plot styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'font.family': 'Arial',
        'figure.autolayout': True
    })

    # Plot 1: Grouped Bar Plot
    if plot_type in ['grouped', 'all']:
        fig, ax = plt.subplots(figsize=(16, 8))
        palette = sns.color_palette("husl", n_colors=len(df['Amino Acid'].unique()))
        
        sns.barplot(
            x='Codon',
            y='RSCU',
            hue='Amino Acid',
            data=df,
            palette=palette,
            dodge=False,
            ax=ax
        )
        
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='RSCU = 1')
        ax.set_title(
            f"Relative Synonymous Codon Usage (RSCU)\nGenetic Code: {genetic_code['name']}",
            pad=20
        )
        ax.set_xlabel("Codons", labelpad=10)
        ax.set_ylabel("RSCU Value", labelpad=10)
        ax.tick_params(axis='x', rotation=90)
        
        ax.legend(
            title='Amino Acid',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0
        )
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_grouped.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_prefix}_grouped.pdf", bbox_inches='tight')
        plt.close()

    # Plot 2: Stacked Bar Plot
    if plot_type in ['stacked', 'all']:
        pivot_data = {}
        for aa in df['Amino Acid'].unique():
            aa_codons = df[df['Amino Acid'] == aa]
            pivot_data[aa] = {}
            for _, row in aa_codons.iterrows():
                pivot_data[aa][row['Codon']] = row['RSCU']

        pivot_df = pd.DataFrame.from_dict(pivot_data, orient='index').fillna(0)
        amino_acids = pivot_df.index
        codons_per_aa = {aa: [col for col in pivot_df.columns if pivot_df.loc[aa, col] > 0]
                        for aa in amino_acids}

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#FFB6C1', '#DAA520', '#87CEEB', '#90EE90', '#D3D3D3']
        color_cycle = colors * (max(len(codons) for codons in codons_per_aa.values()) // len(colors) + 1)

        bottom = pd.Series(0, index=amino_acids)
        for codon_idx in range(max(len(codons) for codons in codons_per_aa.values())):
            values = []
            labels = []
            for aa in amino_acids:
                codons = codons_per_aa[aa]
                if codon_idx < len(codons):
                    codon = codons[codon_idx]
                    rscu = pivot_df.loc[aa, codon]
                    values.append(rscu)
                    labels.append(codon)
                else:
                    values.append(0)
                    labels.append('')

            bars = ax.bar(
                amino_acids,
                values,
                bottom=bottom,
                color=color_cycle[codon_idx],
                width=0.8
            )

            for bar, label in zip(bars, labels):
                if label:
                    height = bar.get_height()
                    if height > 0:
                        bar_idx = int(bar.get_x() + bar.get_width() / 2)
                        aa = amino_acids[bar_idx]
                        y_pos = bottom[aa] + height / 2
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y_pos,
                            label,
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            rotation=0
                        )

            bottom += pd.Series(values, index=amino_acids)

        ax.set_title('Codon Usage', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('RSCU value', labelpad=10)
        ax.set_xticks(range(len(amino_acids)))
        ax.set_xticklabels([AA_1_TO_3[aa] for aa in amino_acids], rotation=45, ha='right')
        ax.set_ylim(0, 6)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_stacked.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_prefix}_stacked.pdf", bbox_inches='tight')
        plt.close()

    # Plot 3: Enhanced Heatmap
    if plot_type in ['heatmap', 'all']:
        # Create pivot table
        pivot_df = df.pivot(index='Amino Acid 3', columns='Codon', values='RSCU').fillna(0)
        
        # Sort by amino acid and codon
        pivot_df = pivot_df[sorted(pivot_df.columns)]
        pivot_df.sort_index(inplace=True)
        
        # Create figure with dynamic sizing
        plt.figure(figsize=(
            max(18, len(pivot_df.columns)*0.7),
            max(10, len(pivot_df.index)*0.8)
        ))
        
        # Create custom colormap
        colors = ["#FFFFFF", "#FFE5E5", "#FFB2B2", "#FF7F7F", "#FF4C4C", "#FF0000"]
        cmap = LinearSegmentedColormap.from_list("custom_reds", colors)
        
        # Plot enhanced heatmap
        ax = sns.heatmap(
            pivot_df,
            cmap=cmap,
            linewidths=0.5,
            linecolor='#DDDDDD',
            annot=True,
            fmt=".1f",
            annot_kws={
                'size': 9,
                'color': 'black',
                'weight': 'normal'
            },
            cbar_kws={
                'label': 'RSCU Value', 
                'shrink': 0.8,
                'aspect': 20
            },
            vmin=0,
            vmax=2.5,
            square=True,
            xticklabels=True,
            yticklabels=True,
            mask=pivot_df == 0
        )
        
        # Highlight preferred codons (RSCU > 1)
        for y in range(pivot_df.shape[0]):
            for x in range(pivot_df.shape[1]):
                if pivot_df.iloc[y, x] > 1.0:
                    ax.add_patch(plt.Rectangle(
                        (x, y), 1, 1,
                        fill=False,
                        edgecolor='blue',
                        lw=1,
                        linestyle='--'
                    ))
        
        # Customize plot
        plt.title(
            f"Codon Usage Bias (RSCU)\n{genetic_code['name']} Genetic Code",
            pad=20,
            fontsize=14,
            weight='bold'
        )
        plt.xlabel('Codon', labelpad=10, fontsize=12)
        plt.ylabel('Amino Acid', labelpad=10, fontsize=12)
        
        # Rotate x-axis labels diagonally
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha='right',
            rotation_mode='anchor',
            fontsize=10
        )
        
        # Adjust y-axis labels
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=10,
            va='center'
        )
        
        # Add grid lines
        ax.set_xticks(np.arange(pivot_df.shape[1]) + 0.5, minor=True)
        ax.set_yticks(np.arange(pivot_df.shape[0]) + 0.5, minor=True)
        ax.grid(which="minor", color="#EEEEEE", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_prefix}_heatmap.pdf", bbox_inches='tight')
        plt.close()

def save_results(rscu, stats, codon_percent, aa_usage, genetic_code_name='standard', output_prefix='rscu'):
    """Save all results to files."""
    genetic_code = get_genetic_code(genetic_code_name)
    try:
        # Save RSCU values
        with open(f"{output_prefix}_values.tsv", 'w') as f:
            f.write("Codon\tAminoAcid\tRSCU\tPercentage\n")
            for codon, value in sorted(rscu.items()):
                aa = genetic_code['table'].get(codon, 'X')
                f.write(f"{codon}\t{aa}\t{value:.4f}\t{codon_percent.get(codon, 0):.2f}%\n")
        
        # Save statistics
        with open(f"{output_prefix}_stats.txt", 'w') as f:
            f.write("=== Codon Usage Statistics ===\n")
            for key, value in stats.items():
                if value is None:
                    value = "N/A"
                elif isinstance(value, float):
                    value = f"{value:.2f}"
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n=== Amino Acid Usage ===\n")
            for aa, count in sorted(aa_usage.items()):
                f.write(f"{aa}: {count} ({count/stats['total_codons']*100:.1f}%)\n")
    except IOError as e:
        raise IOError(f"Error writing output files: {e}")

def load_reference_weights(weights_file):
    """Load reference weights for CAI from a TSV file."""
    if not weights_file:
        return {}
    
    try:
        weights = {}
        with open(weights_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        codon, weight = parts[0], parts[1]
                        weights[codon] = float(weight)
        return weights
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference weights file not found: {weights_file}")
    except ValueError:
        raise ValueError("Invalid format in reference weights file. Expected: codon<TAB>weight")

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Codon Usage Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            Genetic Code Options:
              standard (1)                    - Standard
              vertebrate_mito (2)            - Vertebrate Mitochondrial
              yeast_mito (3)                - Yeast Mitochondrial
              mold_protozoan_mito (4)       - Mold/Protozoan Mitochondrial
              invertebrate_mito (5)          - Invertebrate Mitochondrial
              ciliate (6)                   - Ciliate/Dasycladacean/Hexamita
              echinoderm_flatworm_mito (9)  - Echinoderm/Flatworm Mitochondrial
              euplotid (10)                 - Euplotid Nuclear
              bacterial_plant_plastid (11)  - Bacterial/Archaeal/Plant Plastid
              alternative_yeast (12)        - Alternative Yeast Nuclear
              ascidian_mito (13)            - Ascidian Mitochondrial
              alternative_flatworm_mito (14) - Alternative Flatworm Mitochondrial
              chlorophycean_mito (16)       - Chlorophycean Mitochondrial
              trematode_mito (21)           - Trematode Mitochondrial
              scenedesmus_mito (22)         - Scenedesmus obliquus Mitochondrial
              thraustochytrium_mito (23)    - Thraustochytrium Mitochondrial
              pterobranchia_mito (24)       - Pterobranchia Mitochondrial
            
            Output Files:
              <prefix>_grouped.png/pdf   - RSCU grouped bar plot
              <prefix>_stacked.png/pdf   - RSCU stacked bar plot
              <prefix>_heatmap.png/pdf   - Enhanced RSCU heatmap
              <prefix>_values.tsv       - Tab-separated RSCU values
              <prefix>_stats.txt        - Summary statistics
        ''')
    )
    
    parser.add_argument('input', help="Input FASTA file")
    parser.add_argument('-g', '--genetic_code', default='standard',
                       help="Genetic code to use (default: standard)")
    parser.add_argument('-o', '--output', default='rscu',
                       help="Output prefix (default: rscu)")
    parser.add_argument('-d', '--outdir', default='.',
                       help="Output directory (default: current directory)")
    parser.add_argument('-w', '--weights', help="Reference weights file for CAI (TSV: codon<TAB>weight)")
    parser.add_argument('-m', '--min_length', type=int, default=30,
                       help="Minimum sequence length in nucleotides (default: 30)")
    parser.add_argument('-p', '--parallel', action='store_true',
                       help="Use parallel processing for large datasets")
    parser.add_argument('--plot_type', choices=['grouped', 'stacked', 'heatmap', 'all'], default='all',
                       help="Type of plot to generate (default: all)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    try:
        os.makedirs(args.outdir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Error creating output directory {args.outdir}: {e}")
    
    # Construct full output prefix
    output_prefix = os.path.join(args.outdir, args.output)
    
    # Load reference weights for CAI
    reference_weights = load_reference_weights(args.weights)
    
    # Process sequences
    codon_count, total_codons, seq_lengths, gc_contents = parse_sequences(
        args.input, args.genetic_code, args.min_length, args.parallel
    )
    
    if total_codons == 0:
        print("Error: No valid codons found in input sequences!")
        sys.exit(1)
    
    # Perform analyses
    rscu = calculate_rscu(codon_count, args.genetic_code)
    cai = calculate_cai(codon_count, reference_weights, args.genetic_code)
    stats, codon_percent, aa_usage = generate_stats(
        codon_count, total_codons, seq_lengths, gc_contents, cai, args.genetic_code
    )
    
    # Output results
    plot_rscu(rscu, args.genetic_code, output_prefix, args.plot_type)
    save_results(rscu, stats, codon_percent, aa_usage, args.genetic_code, output_prefix)
    
    # Print summary
    print(f"\nAnalysis complete for {stats['total_sequences']} sequences")
    print(f"Total codons: {stats['total_codons']}")
    print(f"Average GC content: {stats['avg_gc_content']:.2f}%")
    print(f"Effective Number of Codons (ENc): {stats['enc']:.2f}")
    print(f"Codon Adaptation Index (CAI): {stats['cai']:.2f}" if stats['cai'] else "CAI: N/A (no reference weights)")
    print(f"\nResults saved with prefix: {output_prefix}_*")

if __name__ == "__main__":
    main()
