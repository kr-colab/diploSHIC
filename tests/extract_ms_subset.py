#!/usr/bin/env python3
"""Extract a subset of reps from an ms-format output file."""

import argparse
import gzip
import sys


def extract_reps(input_file, output_file, num_reps):
    """Extract first num_reps from an ms-format file."""
    # Determine if input is gzipped
    if input_file.endswith('.gz'):
        open_func = lambda f: gzip.open(f, 'rt')
    else:
        open_func = lambda f: open(f, 'r')

    # Determine if output should be gzipped
    if output_file.endswith('.gz'):
        out_func = lambda f: gzip.open(f, 'wt')
    else:
        out_func = lambda f: open(f, 'w')

    rep_count = 0
    in_rep = False
    header_written = False

    with open_func(input_file) as fin, out_func(output_file) as fout:
        for line in fin:
            # First line is the command/header
            if not header_written:
                # Modify the header to reflect fewer reps
                parts = line.split()
                if len(parts) >= 3:
                    parts[2] = str(num_reps)  # Update number of reps
                fout.write(' '.join(parts) + '\n')
                header_written = True
                continue

            # Track rep boundaries (// marks start of a new rep)
            if line.strip() == '//':
                rep_count += 1
                if rep_count > num_reps:
                    break
                in_rep = True
                fout.write(line)
            elif in_rep:
                fout.write(line)

    print(f"Extracted {min(rep_count, num_reps)} reps to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract subset of reps from ms-format file")
    parser.add_argument("input_file", help="Input ms-format file (can be gzipped)")
    parser.add_argument("output_file", help="Output file (will be gzipped if .gz extension)")
    parser.add_argument("-n", "--num-reps", type=int, default=50,
                        help="Number of reps to extract (default: 50)")
    args = parser.parse_args()

    extract_reps(args.input_file, args.output_file, args.num_reps)


if __name__ == "__main__":
    main()
