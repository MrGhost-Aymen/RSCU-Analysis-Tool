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

# Valid amino acid codes (including stop codon)
VALID_AA_CODES = set('ACDEFGHIKLMNPQRSTVWY*')

# All NCBI genetic codes (25 variants)
GENETIC_CODES = {
    # 1. The Standard Code
    'standard': {
        'name': 'Standard',
        'id': 1,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'CTG', 'ATG']
    },

    # 2. The Vertebrate Mitochondrial Code
    'vertebrate_mito': {
        'name': 'Vertebrate Mitochondrial',
        'id': 2,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': '*', 'AGG': '*',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATT', 'ATC', 'ATA', 'ATG', 'GTG']
    },

    # 3. The Yeast Mitochondrial Code
    'yeast_mito': {
        'name': 'Yeast Mitochondrial',
        'id': 3,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'T', 'CTC': 'T', 'CTA': 'T', 'CTG': 'T',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATA', 'ATG', 'GTG']
    },

    # 4. The Mold, Protozoan, and Coelenterate Mitochondrial Code and the Mycoplasma/Spiroplasma Code
    'mold_protozoan_mito': {
        'name': 'Mold/Protozoan Mitochondrial',
        'id': 4,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'CTG', 'ATG', 'ATT', 'ATC', 'ATA', 'GTG']
    },

    # 5. The Invertebrate Mitochondrial Code
    'invertebrate_mito': {
        'name': 'Invertebrate Mitochondrial',
        'id': 5,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'S', 'AGG': 'S',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'ATG', 'ATT', 'ATC', 'ATA', 'GTG']
    },

    # 6. The Ciliate, Dasycladacean and Hexamita Nuclear Code
    'ciliate': {
        'name': 'Ciliate/Dasycladacean/Hexamita',
        'id': 6,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': 'Q', 'TAG': 'Q',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'CTG', 'ATG']
    },

    # 9. The Echinoderm and Flatworm Mitochondrial Code
    'echinoderm_flatworm_mito': {
        'name': 'Echinoderm/Flatworm Mitochondrial',
        'id': 9,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'N', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'S', 'AGG': 'S',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    },

    # 10. The Euplotid Nuclear Code
    'euplotid': {
        'name': 'Euplotid Nuclear',
        'id': 10,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'C', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'CTG', 'ATG']
    },

    # 11. The Bacterial, Archaeal and Plant Plastid Code
    'bacterial_plant_plastid': {
        'name': 'Bacterial/Archaeal/Plant Plastid',
        'id': 11,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['TTG', 'CTG', 'ATG', 'GTG', 'ATT', 'ATC', 'ATA']
    },

    # 12. The Alternative Yeast Nuclear Code
    'alternative_yeast': {
        'name': 'Alternative Yeast Nuclear',
        'id': 12,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'S',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['CTG', 'ATG']
    },

    # 13. The Ascidian Mitochondrial Code
    'ascidian_mito': {
        'name': 'Ascidian Mitochondrial',
        'id': 13,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'G', 'AGG': 'G',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATA', 'ATG', 'GTG']
    },

    # 14. The Alternative Flatworm Mitochondrial Code
    'alternative_flatworm_mito': {
        'name': 'Alternative Flatworm Mitochondrial',
        'id': 14,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': 'Y', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'N', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'S', 'AGG': 'S',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    },

    # 16. Chlorophycean Mitochondrial Code
    'chlorophycean_mito': {
        'name': 'Chlorophycean Mitochondrial',
        'id': 16,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': 'L',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    },

    # 21. Trematode Mitochondrial Code
    'trematode_mito': {
        'name': 'Trematode Mitochondrial',
        'id': 21,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'N', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'S', 'AGG': 'S',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    },

    # 22. Scenedesmus obliquus Mitochondrial Code
    'scenedesmus_mito': {
        'name': 'Scenedesmus obliquus Mitochondrial',
        'id': 22,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG']
    },

    # 23. Thraustochytrium Mitochondrial Code
    'thraustochytrium_mito': {
        'name': 'Thraustochytrium Mitochondrial',
        'id': 23,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    },

    # 24. Pterobranchia Mitochondrial Code
    'pterobranchia_mito': {
        'name': 'Pterobranchia Mitochondrial',
        'id': 24,
        'table': {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'S', 'AGG': 'S',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        },
        'start': ['ATG', 'GTG']
    }
}

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
                rscu[codon] = observed / expected if expected != 0 else 0.0
    
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

def plot_rscu(rscu, genetic_code_name='standard', output_prefix='rscu'):
    """Create enhanced RSCU visualization with bar plot and heatmap."""
    genetic_code = get_genetic_code(genetic_code_name)
    
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
    df.sort_values(['Amino Acid', 'Codon'], inplace=True)
    
    # Bar plot
    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", n_colors=len(df['Amino Acid'].unique()))
    
    ax = sns.barplot(x='Codon', y='RSCU', hue='Amino Acid', data=df, 
                    palette=palette, dodge=False)
    
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1)
    plt.title(f"Relative Synonymous Codon Usage (RSCU)\nGenetic Code: {genetic_code['name']}")
    plt.xlabel("Codons")
    plt.ylabel("RSCU Value")
    plt.xticks(rotation=90)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='Amino Acid', 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_histogram.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_histogram.pdf", bbox_inches='tight')
    plt.close()
    
    # Heatmap
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index='Amino Acid', columns='Codon', values='RSCU')
    sns.heatmap(pivot_df, cmap="YlOrRd", linewidths=0.5, annot=True, fmt=".2f")
    plt.title(f"RSCU Heatmap by Amino Acid\nGenetic Code: {genetic_code['name']}")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=300)
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
                    codon, weight = line.strip().split()
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
              <prefix>_histogram.png/pdf - RSCU bar plot
              <prefix>_heatmap.png       - RSCU heatmap by AA
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
    plot_rscu(rscu, args.genetic_code, output_prefix)
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