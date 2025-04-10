"""
Develops a one-hot encoding of a nucleotide sequence input matrix and
RNA-seq output matrix describing the start, end, and transition states of
reads within a specified BED window.
"""
# pylint: disable=fixme, disable=no-member

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from Bio import SeqIO
import pyranges as pr
import pysam


class Encoder:
    """
    A class to generate one-hot encodings of nucleotide sequences and transition state matrices
    of RNA-seq data based on reads within specified BED windows.

    Attributes:
    -----------
    bed_file : str
        Path to the BED file specifying genomic regions.
    fasta_file : str
        Path to the FASTA file containing genomic sequences.
    bam_files : list of str
        Paths to BAM files containing RNA-seq read alignments.
    in_memory : Bool
        Boolean describing whether to save output matrices in memory.
    aggregate_bams : Bool
        Boolean describing whether to save aggregate matrices across BAM files.
    bed_windows : pd.DataFrame
        DataFrame holding parsed BED window information.
    onehot_sequence_matrices : list of np.ndarray
        List of one-hot encoded matrices for sequences.
    transition_matrices : list of np.ndarray
        List of transition matrices for read alignments.
    output_dir : str
        Directory to save output files.
    """

    transition_states = [
        ("S", "M"),
        ("M", "M"),
        ("M", "E"),
        ("M", "N"),
        ("N", "M"),
        ("N", "N"),
    ]
    transition_dict = {transition: i for i, transition in enumerate(transition_states)}

    nucleotides = ["A", "C", "G", "T", "N"]

    def __init__(self, bed_file: str, fasta_file: str, bam_files: list[str], output_dir: str = ".",
                 in_memory: bool = False, aggregate_bams: bool = False):
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.bam_files = bam_files
        self.output_dir = output_dir

        self.in_memory = in_memory
        self.aggregate_bams = aggregate_bams

        self.bed_windows = None
        self.onehot_sequence_matrices = None
        self.transition_matrices = None

    # TODO: write some sanity check functions to ensure correct bed/fasta/bam formatting
    # Things to check: bed file formatting (columns, dtypes, etc.), fasta file chromosome names
    # match bam files chromosome names, bed coords greater than zero, etc.
    # also check uniqueness of BED windows

    def extract_bed_windows(self):
        """
        Parses BED file and loads window coordinates into a DataFrame.

        Sets:
        -----
        self.bed_windows : pd.DataFrame
            Contains columns: "Chromosome", "Start", "End" for each BED region.
        """

        self.bed_windows = pr.read_bed(self.bed_file, as_df=True)

        # Sort BED windows by chromosome, coordinates (makes subsequent window extraction more efficient)
        chrom_order = {f"chr{i}": i for i in range(1, 23)}
        chrom_order.update({"chrX": 23, "chrY": 24, "chrM": 25}) # TODO: support non-human chromosome #s
        self.bed_windows["chrom_sort"] = self.bed_windows["Chromosome"].map(chrom_order)
        self.bed_windows = self.bed_windows.sort_values(by=["chrom_sort", "Start", "End"], ascending=True)
        self.bed_windows = self.bed_windows.drop(columns="chrom_sort")

    def extract_sequence_onehot(self):
        """
        Extracts nucleotide sequences for each BED window from a FASTA file and generates one-hot encoded representations.

        Sets
        ----
        self.onehot_sequence_matrices : list of np.ndarray or list of str
            - If self.in_memory is True: stores one-hot encoded matrices for each BED window.
            - If self.in_memory is False: stores file paths to the saved one-hot encoded matrices.
        """

        onehot_sequence_matrices = []
        encoder = OneHotEncoder(categories=[self.nucleotides], dtype=np.uint8, handle_unknown="ignore", sparse_output=False)

        for record in SeqIO.parse(self.fasta_file, "fasta"): # TODO: add a check that all the chromosomes are in this file
            chrom = record.id

            # Filter DataFrame for windows on current chromosome
            chrom_windows = self.bed_windows[self.bed_windows["Chromosome"] == chrom]

            for window in chrom_windows.itertuples():
                start, end = window.Start, window.End
                seq = record.seq[start:end].upper() # TODO: handle case where BED window sequence not found in FASTA file

                # One-hot encode the sequence
                seq_array = np.array(list(seq)).reshape(-1, 1)
                onehot_matrix = encoder.fit_transform(seq_array).T

                if self.in_memory:
                    onehot_sequence_matrices.append(onehot_matrix)
                else:
                    filename = f"sequence_OHE_{chrom}_{start}_{end}.npy"
                    file_loc = os.path.join(self.output_dir, filename)
                    np.save(file_loc, onehot_matrix)
                    onehot_sequence_matrices.append(filename)

        self.onehot_sequence_matrices = onehot_sequence_matrices

    def extract_transition_matrices(self):
        """
        Constructs transition matrices based on read alignments in BAM files for each BED window.

        Sets:
        -----
        self.transition_matrices : list of np.ndarray or list of str
            - If self.in_memory is True: stores transition matrices capturing state changes for reads per BED window.
            - If self.in_memory is False: stores file paths to the saved transition matrices.
        """

        transition_matrices = []

        # TODO: parallelize this!
        for window in self.bed_windows.itertuples():

            chrom, start, end = window.Chromosome, window.Start, window.End
            matrix_shape = (12, end - start)

            if self.aggregate_bams:
                transition_matrix = np.zeros(matrix_shape, dtype=np.uint16)
                for bam_file in self.bam_files:
                    samfile = pysam.AlignmentFile(bam_file, "rb") # TODO: use pysam pileup instead?
                    for read in samfile.fetch(chrom, start, end):
                        self.cigar_transition_features(transition_matrix, start, end, read)

                if self.in_memory:
                    transition_matrices.append(transition_matrix)
                else:
                    filename = f"transitions_{chrom}_{start}_{end}.npy"
                    file_loc = os.path.join(self.output_dir, filename)
                    np.save(file_loc, transition_matrix)
                    transition_matrices.append(filename)

            else: 
                for bam_file in self.bam_files:
                    transition_matrix = np.zeros(matrix_shape, dtype=np.uint16)
                    samfile = pysam.AlignmentFile(bam_file, "rb")
                    for read in samfile.fetch(chrom, start, end):
                        self.cigar_transition_features(transition_matrix, start, end, read)
                    
                    if self.in_memory:
                        transition_matrices.append(transition_matrix)
                    else:
                        filename = f"transitions_{chrom}_{start}_{end}.npy"
                        file_loc = os.path.join(self.output_dir, filename)
                        np.save(file_loc, transition_matrix)
                        transition_matrices.append(filename)

        self.transition_matrices = transition_matrices

    # TODO: better name for this function
    def cigar_transition_features(self, transition_matrix: np.array, window_start: int, window_end: int, read: pysam.AlignedRead):
        """
        Updates transition matrix with CIGAR-based state changes for a given read.

        Parameters:
        -----------
        transition_matrix : np.ndarray
            Matrix tracking state transitions.
        window_start : int
            Start coordinate of the BED window.
        window_end : int
            End coordinate of the BED window.
        read : pysam.AlignedSegment
            BAM alignment data for a given read.
        """

        # Convert pysam's list of CIGAR tuples (run-length encoded splice states) to list of
        # transition states. Also: convert to standard CIGAR codes and ensure cigar_tuples contain
        # only "M" (0) or "N" (3) transition states - TODO: handle other cigar strings?
        cigar_tuples = [("S", "M")]  # TODO: better name for this
        for cigar_tuple in read.cigartuples:
            if cigar_tuple[0] == 0:
                cigar_tuples.extend([("M", cigar_tuple[1] - 1), ("M", "N")])
            elif cigar_tuple[0] == 3:
                cigar_tuples.extend([("N", cigar_tuple[1] - 1), ("N", "M")])
            else:
                return
        cigar_tuples[-1] = ("M", "E")

        # Line up read to BED window
        read_offset = read.reference_start - window_start
        cigar_tuples, read_offset = self.trim_cigar_tuples(cigar_tuples, read_offset)

        # Add transition features as described by cigar_tuples
        reverse_offset = 6 if read.is_reverse else 0
        features_added = 0
        for cigar_tuple in cigar_tuples:
            # TODO: improve this code somewhat
            if not isinstance(cigar_tuple[1], int): # if non-self-loop transition (new CIGAR state)
                transition_feat = self.transition_dict[(cigar_tuple[0], cigar_tuple[1])]
                if window_start + read_offset + features_added < window_end:
                    transition_matrix[transition_feat + reverse_offset][read_offset + features_added] += 1
                else:
                    return
                features_added += 1

            else: # self-loop transition (repetition of CIGAR state)
                transition_feat = self.transition_dict[(cigar_tuple[0], cigar_tuple[0])]
                for _ in range(cigar_tuple[1]):
                    if window_start + read_offset + features_added < window_end:
                        transition_matrix[transition_feat + reverse_offset][read_offset + features_added] += 1
                    else:
                        return
                    features_added += 1

    def trim_cigar_tuples(self, cigar_tuples: list[tuple[str, str | int]], read_offset: int) -> tuple[list[tuple[str, str | int]], int]:
        """
        Adjusts CIGAR tuples if the read starts before the BED window.

        Parameters:
        -----------
        cigar_tuples : list of tuple of (str, str | int)
            CIGAR tuples corresponding to a particular read.
        read_offset : int
            Offset of read alignment relative to the window start.

        Returns:
        --------
        list of tuple of (str, str | int)
            Adjusted CIGAR tuples.
        int
            Updated offset of read alignment corresponding to adjusted CIGAR tuples.
        """

        # TODO: improve this code somewhat
        # "Fast-forward" the CIGAR tuple encoding of reads that begin before the BED window
        i = 0  # Keep track of count of processed tuples, to be sliced out later
        while read_offset < 0:
            repetition = cigar_tuples[i][1]
            if not isinstance(repetition, int): # if non-self-loop transition (new CIGAR state)
                read_offset += 1
                i += 1
                continue
            if read_offset + repetition <= 0:
                read_offset += repetition
                i += 1
                continue
            cigar_tuples[i] = (cigar_tuples[i][0], cigar_tuples[i][1] + read_offset)
            read_offset = 0

        return cigar_tuples[i:], read_offset

    def encode(self):
        """Driver method to orchestrate encoding processes."""
        self.extract_bed_windows()
        self.extract_sequence_onehot()
        self.extract_transition_matrices()

    def save_matrices(self):
        """
        Save one-hot sequence and transition matrices to files in the output directory.
        """

        os.makedirs(self.output_dir, exist_ok=True)

        for i, (seq_matrix, trans_matrix) in enumerate(zip(self.onehot_sequence_matrices, self.transition_matrices)):
            # Save one-hot sequence matrix
            seq_filename = os.path.join(self.output_dir, f"onehot_sequence_matrix_{i}.npy")
            np.save(seq_filename, seq_matrix)

            # Save transition matrix
            trans_filename = os.path.join(self.output_dir, f"transition_matrix_{i}.npy")
            np.save(trans_filename, trans_matrix)

            print(f"Saved matrices for window {i}:")
            print(f"  Sequence matrix: {seq_filename}")
            print(f"  Transition matrix: {trans_filename}")

    def save_matrix_as_bedgraph(self):
        """
        Save one-hot sequence and transition matrices to files in the output directory in text form.
        """
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        for i, (seq_matrix, trans_matrix, bed_window) in enumerate(
            zip(
                self.onehot_sequence_matrices,
                self.transition_matrices,
                self.bed_windows.itertuples(),
            )
        ):
            # Save one-hot sequence matrix
            seq_filename = os.path.join(
                self.output_dir, f"onehot_sequence_matrix_{i}.txt"
            )
            np.savetxt(seq_filename, seq_matrix)

            # Save transition matrix
            trans_filename = os.path.join(self.output_dir, f"transition_matrix_{i}.txt")
            np.savetxt(trans_filename, trans_matrix)

            print(f"Saved matrices for window {i}:")
            print(f"  Sequence matrix: {seq_filename}")
            print(f"  Transition matrix: {trans_filename}")
            df_filename = os.path.join(
                self.output_dir, f"transitions_matrix_${i}.bedgraph"
            )
            trans_df = pd.DataFrame(data=trans_matrix.transpose())
            trans_df.insert(0, "chrom", bed_window[1])
            trans_df.insert(
                1, "start", range(bed_window[2], bed_window[2] + trans_df.shape[0])
            )
            trans_df.insert(
                2,
                "end",
                range(bed_window[2] + 1, bed_window[2] + trans_df.shape[0] + 1),
            )
            trans_df.to_csv(df_filename, sep="\t", index=False, header=False)


def main():
    """Parse command-line arguments and run encoding process."""
    
    parser = argparse.ArgumentParser(description="One-hot encode genomic sequences and RNA-seq reads.")

    parser.add_argument("-b", "--bed", required=True, help="Path to one or more BED files")
    parser.add_argument("-f", "--fasta", required=True, help="Path to FASTA file containing genomic sequences")
    parser.add_argument("-a", "--bam", nargs="+", required=True, help="Path to one or more BAM files")

    parser.add_argument("--in_memory", action=argparse.BooleanOptionalAction, help="Save OHE sequence and transition matrices in memory")
    parser.add_argument("--aggregate_bams", action=argparse.BooleanOptionalAction, help="Aggregate transition matrices across BAM files")

    parser.add_argument("-o", "--output", default=".", help="Output directory for saving matrices (default: current directory)")
    parser.add_argument("-p", "--print", action="store_true", help="Print matrices to console in addition to saving")

    args = parser.parse_args()

    # Create Encoder instance and run encoding process
    encoder = Encoder(bed_file=args.bed, fasta_file=args.fasta, bam_files=args.bam, output_dir=args.output,
                      in_memory=args.in_memory, aggregate_bams=args.aggregate_bams)
    encoder.encode()

    # Save matrices
    if args.in_memory:
        encoder.save_matrices()
        encoder.save_matrix_as_bedgraph()

    # Optionally print matrices to console
    if args.print:
        for i, matrices in enumerate(zip(encoder.onehot_sequence_matrices, encoder.transition_matrices)):
            print(f"\nOne-Hot Sequence Matrix {i}:\n{matrices[0]}\n")
            print(f"\nTransition Matrix {i}:\n{matrices[1]}\n")

if __name__ == "__main__":
    main()
