
# RSCU Analysis Tool

The **RSCU Analysis Tool** (`RSCU.py`) is a Python script for analyzing codon usage in DNA sequences. It calculates **Relative Synonymous Codon Usage (RSCU)**, **Codon Adaptation Index (CAI)**, **Effective Number of Codons (ENc)**, and **GC content** from FASTA-formatted coding sequences. The tool supports 25 NCBI genetic codes, including standard, mitochondrial, and plastid codes, and provides visualizations (bar plots and heatmaps) and detailed output files.

## Features

- **Codon Usage Analysis**: Computes RSCU for each codon, showing bias in synonymous codon usage.
- **Multiple Genetic Codes**: Supports 25 NCBI genetic codes, including `standard`, `bacterial_plant_plastid`, `chlorophycean_mito`, and more.
- **Advanced Metrics**:
  - **CAI**: Measures codon adaptation to a reference set (optional).
  - **ENc**: Quantifies codon usage bias.
  - **GC Content**: Calculates average GC percentage across sequences.
- **Parallel Processing**: Speeds up analysis for large datasets using multiprocessing.
- **Visualizations**:
  - Bar plot of RSCU values by codon.
  - Heatmap of RSCU by amino acid.
- **Comprehensive Outputs**: Generates TSV files, text summaries, and plots (PNG and PDF formats).
- **Input Validation**: Checks for valid codons, sequence lengths, and genetic code compatibility.

## Installation

### Prerequisites

- **Python 3.6+**
- Required Python packages:
  - `biopython`
  - `matplotlib`
  - `seaborn`
  - `pandas`
  - `numpy`
  - `scipy`

### Install Dependencies

Install the required packages using pip:

```bash
pip install biopython matplotlib seaborn pandas numpy scipy
```

### Download the Script

Clone the repository or download `RSCU.py`:

```bash
git clone https://github.com/yourusername/rscu-analysis.git
cd rscu-analysis
```

## Usage

Run the script from the command line, providing a FASTA file and optional arguments.

### Basic Command

```bash
python RSCU.py input.fasta -g bacterial_plant_plastid
```

### Full Command with Options

```bash
python RSCU.py input.fasta -g bacterial_plant_plastid -o output -d results -w weights.tsv -m 30 -p
```

### Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `input` | Input FASTA file (required) | \- |
| `-g, --genetic_code` | Genetic code to use (e.g., `standard`, `bacterial_plant_plastid`) | `standard` |
| `-o, --output` | Output file prefix | `rscu` |
| `-d, --outdir` | Output directory | `.` (current directory) |
| `-w, --weights` | TSV file with CAI reference weights (format: `codon<TAB>weight`) | None |
| `-m, --min_length` | Minimum sequence length (nucleotides) | `30` |
| `-p, --parallel` | Enable parallel processing for large datasets | Disabled |

### Supported Genetic Codes

The tool supports 25 NCBI genetic codes, including:

- `standard` (1)
- `vertebrate_mito` (2)
- `yeast_mito` (3)
- `mold_protozoan_mito` (4)
- `invertebrate_mito` (5)
- `ciliate` (6)
- `echinoderm_flatworm_mito` (9)
- `euplotid` (10)
- `bacterial_plant_plastid` (11)
- `alternative_yeast` (12)
- `ascidian_mito` (13)
- `alternative_flatworm_mito` (14)
- `chlorophycean_mito` (16)
- `trematode_mito` (21)
- `scenedesmus_mito` (22)
- `thraustochytrium_mito` (23)
- `pterobranchia_mito` (24)

Run `python RSCU.py -h` to see the full list in the help message.

## Input

- **FASTA File**: A file containing DNA coding sequences (CDS) in FASTA format.

  - Sequences should be multiples of 3 nucleotides (codons).
  - Non-coding sequences or sequences with ambiguous bases (e.g., `N`) may be skipped or flagged.
  - Example:

    ```
    >sequence1
    ATGAAACTTGGTTAA
    >sequence2
    ATGTCCTTAGCTTAA
    ```

- **Weights File (Optional)**: A TSV file for CAI calculation, with codon weights.

  - Format: One codon and weight per line, separated by a tab.
  - Example:

    ```
    AAA	1.0
    AAG	0.8
    TTT	0.9
    ```

## Output

The script generates the following files in the specified output directory (default: current directory):

| File | Description |
| --- | --- |
| `<prefix>_histogram.png` | Bar plot of RSCU values by codon |
| `<prefix>_histogram.pdf` | PDF version of the RSCU bar plot |
| `<prefix>_heatmap.png` | Heatmap of RSCU values by amino acid |
| `<prefix>_values.tsv` | TSV file with codon, amino acid, RSCU, and percentage |
| `<prefix>_stats.txt` | Text file with summary statistics (sequences, codons, GC content, ENc, CAI) |

### Example Output

For input `ndhF.fasta` with prefix `output`:

- `output_histogram.png`: RSCU bar plot.
- `output_values.tsv`:

  ```
  Codon	AminoAcid	RSCU	Percentage
  AAA	K	1.2000	2.50%
  AAG	K	0.8000	1.67%
  ...
  ```
- `output_stats.txt`:

  ```
  === Codon Usage Statistics ===
  Total Sequences: 10
  Total Codons: 5000
  Average Sequence Length: 500.0
  Median Sequence Length: 500.0
  Average GC Content: 45.2
  Enc: 48.3
  Cai: N/A
  === Amino Acid Usage ===
  A: 400 (8.0%)
  C: 150 (3.0%)
  ...
  ```
![rscu_stacked](https://github.com/user-attachments/assets/c0f5427b-d798-4f01-b31c-9edc687fcf29)
![rscu_heatmap](https://github.com/user-attachments/assets/f70cc745-18f2-438e-baf5-3df75be7e26f)
![rscu_grouped](https://github.com/user-attachments/assets/d3088a0d-4c8e-45ee-87e8-e0514ac5b81c)

### Console Output

A summary is printed to the console:

```
Analysis complete for 10 sequences
Total codons: 5000
Average GC content: 45.20%
Effective Number of Codons (ENc): 48.30
Codon Adaptation Index (CAI): N/A (no reference weights)
Results saved with prefix: results/output_*
```

## Example

Analyze the `ndhF` gene sequences with the `bacterial_plant_plastid` genetic code:

```bash
python RSCU.py ndhF.fasta -g bacterial_plant_plastid -o ndhf_analysis -d results
```

With CAI weights and parallel processing:

```bash
python RSCU.py ndhF.fasta -g bacterial_plant_plastid -o ndhf_analysis -d results -w weights.tsv -p
```

## Troubleshooting

- **FileNotFoundError**: Ensure the input FASTA file and weights file (if provided) are in the correct path.
- **No valid codons found**: Check that sequences are valid CDS (multiples of 3, no ambiguous bases).
- **Invalid genetic code**: Verify the genetic code name matches one of the supported codes (run `python RSCU.py -h`).
- **Missing dependencies**: Install all required packages using `pip`.
- **Syntax errors**: Ensure the script is not corrupted (download a fresh copy from the repository).

For specific issues, open an issue on GitHub or provide details about the error and input file.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please include tests and update documentation for new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, open an issue on GitHub or contact the maintainer at ouamoa@gmail.com.

---

*Last updated: May 4, 2025*
