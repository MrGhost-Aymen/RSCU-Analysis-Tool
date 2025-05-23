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
import logging
import plotly.express as px
from prince import CA
import openpyxl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('codon_usage.log'),
        logging.StreamHandler()
    ]
)

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

class CodonUsageAnalyzer:
    def __init__(self, fasta_file, genetic_code='standard', min_length=30, parallel=False):
        self.fasta_file = fasta_file
        self.genetic_code_name = genetic_code
        self.min_length = min_length
        self.parallel = parallel
        with open('genetic_codes.json', 'r') as f:
            self.GENETIC_CODES = json.load(f)
        self.validate_genetic_codes()
        self.genetic_code = self.get_genetic_code()
        self.codon_count = defaultdict(int)
        self.total_codons = 0
        self.seq_lengths = []
        self.gc_contents = []
        self.gc1_contents = []
        self.gc2_contents = []
        self.gc3_contents = []
        self.invalid_codons = set()
        self.gene_level_data = []

    def validate_genetic_codes(self):
        """Validate all genetic code tables for correct amino acid codes."""
        for code_name, code in self.GENETIC_CODES.items():
            invalid_aas = [aa for aa in code['table'].values() if aa not in VALID_AA_CODES]
            if invalid_aas:
                raise ValueError(
                    f"Invalid amino acid codes in {code_name}: {', '.join(invalid_aas)}. "
                    f"Valid codes: {', '.join(sorted(VALID_AA_CODES))}"
                )
            if not code['start']:
                raise ValueError(f"No start codons defined for {code_name}")

    def get_genetic_code(self):
        """Return the specified genetic code dictionary."""
        if self.genetic_code_name not in self.GENETIC_CODES:
            raise ValueError(
                f"Invalid genetic code: {self.genetic_code_name}. Available codes: {', '.join(self.GENETIC_CODES.keys())}"
            )
        return self.GENETIC_CODES.get(self.genetic_code_name, self.GENETIC_CODES['standard'])

    def reverse_table(self):
        """Create amino acid to codons mapping, excluding stop codons."""
        aa_to_codons = defaultdict(list)
        for codon, aa in self.genetic_code['table'].items():
            if aa != '*':
                aa_to_codons[aa].append(codon)
        return aa_to_codons

    def calculate_gc_content(self, seq):
        """Calculate GC content of a sequence."""
        seq = seq.upper()
        gc_count = seq.count('G') + seq.count('C')
        total = len(seq)
        return (gc_count / total * 100) if total > 0 else 0

    def process_sequence(self, record):
        """Process a single sequence for codon counting and GC content."""
        codon_count = defaultdict(int)
        invalid_codons = set()
        seq = str(record.seq).upper()
        if 'U' in seq:  # Convert RNA to DNA
            seq = seq.replace('U', 'T')

        # Check sequence length
        if len(seq) < self.min_length:
            return None, None, 0, set(), (0, 0, 0)

        if len(seq) % 3 != 0:
            logging.warning(f"Sequence {record.id} length {len(seq)} is not a multiple of 3")
            return None, None, 0, set(), (0, 0, 0)

        seq_length = len(seq) // 3
        gc_content = self.calculate_gc_content(seq)

        gc1_count = gc2_count = gc3_count = 0
        total_codons_in_seq = 0

        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in self.genetic_code['table']:
                aa = self.genetic_code['table'][codon]
                if aa != '*':
                    codon_count[codon] += 1
                    gc1_count += 1 if codon[0] in 'GC' else 0
                    gc2_count += 1 if codon[1] in 'GC' else 0
                    gc3_count += 1 if codon[2] in 'GC' else 0
                    total_codons_in_seq += 1
            else:
                invalid_codons.add(codon)

        gc1 = (gc1_count / total_codons_in_seq * 100) if total_codons_in_seq > 0 else 0
        gc2 = (gc2_count / total_codons_in_seq * 100) if total_codons_in_seq > 0 else 0
        gc3 = (gc3_count / total_codons_in_seq * 100) if total_codons_in_seq > 0 else 0

        # Calculate ENc for this sequence
        enc = self.calculate_enc(codon_count)

        return codon_count, gc_content, seq_length, invalid_codons, (gc1, gc2, gc3), enc

    def parse_sequences(self):
        """Parse sequences and count codons, optionally in parallel."""
        try:
            sequences = list(SeqIO.parse(self.fasta_file, "fasta"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {self.fasta_file}")

        if not sequences:
            raise ValueError("No sequences found in input file")

        if self.parallel:
            with Pool() as pool:
                results = pool.starmap(
                    self.process_sequence,
                    [(record,) for record in sequences]
                )
        else:
            results = [self.process_sequence(record) for record in sequences]

        for idx, result in enumerate(results):
            if result[0] is None:  # Skipped sequence
                continue
            seq_codon_count, gc_content, seq_length, seq_invalid_codons, (gc1, gc2, gc3), enc = result
            for codon, count in seq_codon_count.items():
                self.codon_count[codon] += count
                self.total_codons += count
            self.seq_lengths.append(seq_length)
            self.gc_contents.append(gc_content)
            self.gc1_contents.append(gc1)
            self.gc2_contents.append(gc2)
            self.gc3_contents.append(gc3)
            self.invalid_codons.update(seq_invalid_codons)
            self.gene_level_data.append({
                'gene_id': sequences[idx].id,
                'enc': enc,
                'gc3': gc3,
                'codon_counts': seq_codon_count
            })

        if self.invalid_codons:
            logging.warning(f"Found {len(self.invalid_codons)} invalid codons: {', '.join(sorted(self.invalid_codons))}")

    def calculate_rscu(self):
        """Calculate Relative Synonymous Codon Usage (RSCU)."""
        aa_to_codons = self.reverse_table()
        rscu = {}

        for aa, codons in aa_to_codons.items():
            total = sum(self.codon_count.get(codon, 0) for codon in codons)
            n = len(codons)

            if total == 0:
                for codon in codons:
                    rscu[codon] = 0.0
            else:
                for codon in codons:
                    observed = self.codon_count.get(codon, 0)
                    expected = total / n
                    rscu[codon] = min(observed / expected if expected != 0 else 0.0, 2.5)  # Cap RSCU at 2.5

        return rscu

    def calculate_cai(self, reference_weights):
        """Calculate Codon Adaptation Index (CAI) using reference weights."""
        if not reference_weights:
            logging.warning("No reference weights provided for CAI calculation")
            return None

        cai = 0
        total = 0

        for codon, count in self.codon_count.items():
            aa = self.genetic_code['table'].get(codon, 'X')
            if aa != '*' and codon in reference_weights:
                weight = reference_weights.get(codon, 1.0)
                if weight > 0:  # Avoid log(0)
                    cai += count * np.log(weight)
                    total += count

        return np.exp(cai / total) if total > 0 else None

    def calculate_enc(self, codon_count):
        """Calculate Effective Number of Codons (ENc)."""
        aa_to_codons = self.reverse_table()
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

    def generate_stats(self, cai):
        """Generate comprehensive statistics including GC content."""
        aa_to_codons = self.reverse_table()

        stats = {
            'total_sequences': len(self.seq_lengths),
            'total_codons': self.total_codons,
            'avg_seq_length': np.mean(self.seq_lengths) if self.seq_lengths else 0,
            'median_seq_length': np.median(self.seq_lengths) if self.seq_lengths else 0,
            'avg_gc_content': np.mean(self.gc_contents) if self.gc_contents else 0,
            'avg_gc1_content': np.mean(self.gc1_contents) if self.gc1_contents else 0,
            'avg_gc2_content': np.mean(self.gc2_contents) if self.gc2_contents else 0,
            'avg_gc3_content': np.mean(self.gc3_contents) if self.gc3_contents else 0,
            'enc': self.calculate_enc(self.codon_count),
            'cai': cai
        }

        codon_percent = {codon: (count / self.total_codons * 100) if self.total_codons else 0
                         for codon, count in self.codon_count.items()}

        aa_usage = defaultdict(int)
        for codon, count in self.codon_count.items():
            aa = self.genetic_code['table'].get(codon, 'X')
            aa_usage[aa] += count

        return stats, codon_percent, aa_usage

    def plot_rscu(self, rscu, output_prefix, plot_type='all'):
        """Create multiple visualizations of RSCU values."""
        data = []
        for codon, value in rscu.items():
            aa = self.genetic_code['table'].get(codon, 'X')
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
                f"Relative Synonymous Codon Usage (RSCU)\nGenetic Code: {self.genetic_code['name']}",
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
            pivot_df = df.pivot(index='Amino Acid 3', columns='Codon', values='RSCU').fillna(0)
            pivot_df = pivot_df[sorted(pivot_df.columns)]
            pivot_df.sort_index(inplace=True)

            plt.figure(figsize=(
                max(18, len(pivot_df.columns)*0.7),
                max(10, len(pivot_df.index)*0.8)
            ))

            colors = ["#FFFFFF", "#FFE5E5", "#FFB2B2", "#FF7F7F", "#FF4C4C", "#FF0000"]
            cmap = LinearSegmentedColormap.from_list("custom_reds", colors)

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

            plt.title(
                f"Codon Usage Bias (RSCU)\n{self.genetic_code['name']} Genetic Code",
                pad=20,
                fontsize=14,
                weight='bold'
            )

            plt.xlabel('Codon', labelpad=10, fontsize=12)
            plt.ylabel('Amino Acid', labelpad=10, fontsize=12)

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                ha='right',
                rotation_mode='anchor',
                fontsize=10
            )

            ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=10,
                va='center'
            )

            ax.set_xticks(np.arange(pivot_df.shape[1]) + 0.5, minor=True)
            ax.set_yticks(np.arange(pivot_df.shape[0]) + 0.5, minor=True)
            ax.grid(which="minor", color="#EEEEEE", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            plt.tight_layout()
            plt.savefig(f"{output_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_prefix}_heatmap.pdf", bbox_inches='tight')
            plt.close()

        # Interactive Plotly Plot
        if plot_type in ['grouped', 'all']:
            fig = px.bar(
                df,
                x='Codon',
                y='RSCU',
                color='Amino Acid 3',
                hover_data=['Codon', 'RSCU', 'Amino Acid'],
                title=f"Interactive RSCU Plot (Genetic Code: {self.genetic_code['name']})"
            )
            fig.add_hline(y=1.0, line_dash="dot", line_color="red")
            fig.update_layout(
                xaxis_title="Codons",
                yaxis_title="RSCU Value",
                legend_title="Amino Acid"
            )
            fig.write_html(f"{output_prefix}_interactive.html")

    def plot_enc_gc3(self, output_prefix):
        """Plots ENc vs GC3 for each gene."""
        df = pd.DataFrame(self.gene_level_data)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='gc3', y='enc', data=df, alpha=0.6, label='Genes')

        gc3_vals = np.linspace(0, 100, 200)
        s = gc3_vals / 100.0
        expected_enc = 2 + s + (1 / (s**2 + (1-s)**2))

        plt.plot(gc3_vals, expected_enc, color='red', linestyle='--', label='Expected (Mutation Pressure)')

        plt.xlim(0, 100)
        plt.ylim(20, 65)
        plt.xlabel("GC Content at 3rd Codon Position (GC3s) %")
        plt.ylabel("Effective Number of Codons (ENc)")
        plt.title("ENc vs. GC3 Plot")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"{output_prefix}_enc_vs_gc3.png", dpi=300)
        plt.close()

    def plot_ca(self, output_prefix):
        """Perform Correspondence Analysis and plot first two components."""
        codon_counts = [d['codon_counts'] for d in self.gene_level_data]
        gene_ids = [d['gene_id'] for d in self.gene_level_data]
        if not codon_counts:
            logging.warning("No gene-level data for Correspondence Analysis")
            return

        df = pd.DataFrame(codon_counts, index=gene_ids).fillna(0)
        ca = CA(n_components=2)
        ca = ca.fit(df)

        coords = ca.row_coordinates(df)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=coords[0], y=coords[1], alpha=0.6)

        plt.xlabel("CA1")
        plt.ylabel("CA2")
        plt.title("Correspondence Analysis of Codon Usage")
        plt.grid(True)

        plt.savefig(f"{output_prefix}_ca.png", dpi=300)
        plt.close()

    def save_results(self, rscu, stats, codon_percent, aa_usage, output_prefix):
        """Save all results to files."""
        try:
            # Prepare dataframes for Excel output
            rscu_df = pd.DataFrame([
                {
                    'Codon': codon,
                    'AminoAcid': self.genetic_code['table'].get(codon, 'X'),
                    'RSCU': value,
                    'Percentage': f"{codon_percent.get(codon, 0):.2f}%"
                }
                for codon, value in sorted(rscu.items())
            ])

            stats_df = pd.DataFrame(
                [(k.replace('_', ' ').title(), v if isinstance(v, str) else f"{v:.2f}" if v is not None else "N/A")
                 for k, v in stats.items()],
                columns=['Metric', 'Value']
            )

            aa_usage_df = pd.DataFrame(
                [(aa, count, f"{count/stats['total_codons']*100:.1f}%")
                 for aa, count in sorted(aa_usage.items())],
                columns=['Amino Acid', 'Count', 'Percentage']
            )

            # Save to Excel
            with pd.ExcelWriter(f"{output_prefix}_results.xlsx") as writer:
                rscu_df.to_excel(writer, sheet_name='RSCU_Values', index=False)
                stats_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                aa_usage_df.to_excel(writer, sheet_name='Amino_Acid_Usage', index=False)

            # Save RSCU values (TSV for compatibility)
            with open(f"{output_prefix}_values.tsv", 'w') as f:
                f.write("Codon\tAminoAcid\tRSCU\tPercentage\n")
                for codon, value in sorted(rscu.items()):
                    aa = self.genetic_code['table'].get(codon, 'X')
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

    def load_reference_weights(self, weights_file):
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

    def calculate_all_metrics(self, weights_file=None):
        """Calculate all metrics and store results."""
        self.parse_sequences()
        if self.total_codons == 0:
            raise ValueError("No valid codons found in input sequences")
        reference_weights = self.load_reference_weights(weights_file)
        self.rscu = self.calculate_rscu()
        self.cai = self.calculate_cai(reference_weights)
        self.stats, self.codon_percent, self.aa_usage = self.generate_stats(self.cai)

    def generate_plots(self, output_prefix, plot_type='all'):
        """Generate all plots."""
        self.plot_rscu(self.rscu, output_prefix, plot_type)
        self.plot_enc_gc3(output_prefix)
        self.plot_ca(output_prefix)

    def save_results_wrapper(self, output_prefix):
        """Wrapper to save results."""
        self.save_results(self.rscu, self.stats, self.codon_percent, self.aa_usage, output_prefix)

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
              <prefix>_enc_vs_gc3.png   - ENc vs GC3 plot
              <prefix>_ca.png           - Correspondence Analysis plot
              <prefix>_interactive.html - Interactive RSCU plot
              <prefix>_values.tsv       - Tab-separated RSCU values
              <prefix>_stats.txt        - Summary statistics
              <prefix>_results.xlsx     - Excel file with all results
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

    try:
        os.makedirs(args.outdir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Error creating output directory {args.outdir}: {e}")

    output_prefix = os.path.join(args.outdir, args.output)
    analyzer = CodonUsageAnalyzer(args.input, args.genetic_code, args.min_length, args.parallel)
    analyzer.calculate_all_metrics(args.weights)
    analyzer.generate_plots(output_prefix, args.plot_type)
    analyzer.save_results_wrapper(output_prefix)

    logging.info(f"Analysis complete for {analyzer.stats['total_sequences']} sequences")
    logging.info(f"Total codons: {analyzer.stats['total_codons']}")
    logging.info(f"Average GC content: {analyzer.stats['avg_gc_content']:.2f}%")
    logging.info(f"Average GC1 content: {analyzer.stats['avg_gc1_content']:.2f}%")
    logging.info(f"Average GC2 content: {analyzer.stats['avg_gc2_content']:.2f}%")
    logging.info(f"Average GC3 content: {analyzer.stats['avg_gc3_content']:.2f}%")
    logging.info(f"Effective Number of Codons (ENc): {analyzer.stats['enc']:.2f}")
    logging.info(f"Codon Adaptation Index (CAI): {analyzer.stats['cai']:.2f}" if analyzer.stats['cai'] else "CAI: N/A (no reference weights)")
    logging.info(f"Results saved with prefix: {output_prefix}_*")

if __name__ == "__main__":
    main()
