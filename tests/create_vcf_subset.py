#!/usr/bin/env python3
"""Create a smaller VCF subset for regression testing."""

import gzip
import sys

def create_vcf_subset(input_vcf, output_vcf, num_samples=50,
                      start_pos=28000000, end_pos=28110000):
    """
    Extract a subset of samples and positions from a VCF file.

    Args:
        input_vcf: Path to input VCF (can be gzipped)
        output_vcf: Path to output VCF (will be gzipped if .gz extension)
        num_samples: Number of samples to keep (first N samples)
        start_pos: Start position to include
        end_pos: End position to include
    """
    # Determine open functions based on file extensions
    if input_vcf.endswith('.gz'):
        in_open = lambda f: gzip.open(f, 'rt')
    else:
        in_open = lambda f: open(f, 'r')

    if output_vcf.endswith('.gz'):
        out_open = lambda f: gzip.open(f, 'wt')
    else:
        out_open = lambda f: open(f, 'w')

    snp_count = 0
    with in_open(input_vcf) as fin, out_open(output_vcf) as fout:
        for line in fin:
            # Handle header lines
            if line.startswith('##'):
                fout.write(line)
                continue

            # Handle column header line - subset samples
            if line.startswith('#CHROM'):
                parts = line.strip().split('\t')
                # Keep first 9 fixed columns + first num_samples samples
                subset_parts = parts[:9] + parts[9:9+num_samples]
                fout.write('\t'.join(subset_parts) + '\n')
                print(f"Keeping {num_samples} samples (of {len(parts)-9} total)")
                continue

            # Handle data lines - filter by position and subset samples
            parts = line.strip().split('\t')
            pos = int(parts[1])

            if start_pos <= pos <= end_pos:
                # Keep first 9 fixed columns + first num_samples genotypes
                subset_parts = parts[:9] + parts[9:9+num_samples]
                fout.write('\t'.join(subset_parts) + '\n')
                snp_count += 1

    print(f"Wrote {snp_count} SNPs in region {start_pos}-{end_pos}")
    print(f"Output: {output_vcf}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create VCF subset for testing")
    parser.add_argument("input_vcf", help="Input VCF file")
    parser.add_argument("output_vcf", help="Output VCF file")
    parser.add_argument("-n", "--num-samples", type=int, default=50,
                        help="Number of samples to keep (default: 50)")
    parser.add_argument("-s", "--start", type=int, default=28000000,
                        help="Start position (default: 28000000)")
    parser.add_argument("-e", "--end", type=int, default=28110000,
                        help="End position (default: 28110000)")

    args = parser.parse_args()

    create_vcf_subset(args.input_vcf, args.output_vcf,
                      args.num_samples, args.start, args.end)
