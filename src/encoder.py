"""
Develops a one-hot encoding of a nucleotide sequence input matrix and
RNA-seq output matrix describing the start, end, and transition states of
reads within a specified BED window.
"""
# pylint: disable=fixme, disable=no-member

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from Bio import SeqIO
import pyranges as pr
import pysam


class Encoder:
    """
    A class to generate one-hot encodings of nucleotide sequences and
    transition state matrices of RNA-seq data based on reads within specified
    BED windows.

    Attributes:
    -----------
    bed_files : list of str
        Paths to BED files specifying genomic regions.
    fasta_file : str
        Path to the FASTA file containing genomic sequences.
    bam_files : list of str
        Paths to BAM files containing RNA-seq read alignments.
    bed_windows : pd.DataFrame
        DataFrame holding parsed BED window information.
    onehot_sequence_matrices : list of np.ndarray
        List of one-hot encoded matrices for sequences.
    transition_matrices : list of np.ndarray
        List of transition matrices for read alignments.
    output_dir : str
        Directory to save output files.
    """

    transition_states = [('S','M'), ('M','M'), ('M','E'), ('M','N'), ('N','M'), ('N','N')]
    transition_dict = {transition: i for i, transition in enumerate(transition_states)}

    def __init__(self, bed_files: list[str], fasta_file: str, bam_files: list[str], output_dir: str = '.'):
        self.bed_files = bed_files
        self.fasta_file = fasta_file
        self.bam_files = bam_files
        self.output_dir = output_dir

        self.bed_windows = None
        self.onehot_sequence_matrices = None
        self.transition_matrices = None


    def extract_bed_windows(self):
        """
        Parses BED files and loads window coordinates into a DataFrame.

        Sets:
        -----
        self.bed_windows : pd.DataFrame
            Contains columns: 'chrom', 'start', 'end' for each BED region.
        """

        # Parse BED files to generate list of BED windows
        bed_data = []
        for bed_file in self.bed_files:
            gr = pr.read_bed(bed_file)
            for window in gr.df.itertuples():
                # TODO: include interval.name?, handle repeat windows?
                data = (window.Chromosome, window.Start, window.End)
                bed_data.append(data)

        # Create DataFrame of BED windows - TODO: switch this to a list of bedparse Interval objects?
        self.bed_windows = pd.DataFrame(bed_data, columns=['chrom', 'start', 'end'])

    def extract_sequence_onehot(self):
        """
        Extracts nucleotide sequences from FASTA file for each BED window and
        generates one-hot encodings.

        Sets:
        -----
        self.onehot_sequence_matrices : list of np.ndarray
            One-hot encoded matrices for each BED window's sequence.
        """

        onehot_sequences_matrices = [] # TODO: convert to numpy array?
        categories = [['A', 'T', 'C', 'G']]
        encoder = OneHotEncoder(handle_unknown='ignore', categories=categories, sparse_output=False)

        # Parse FASTA file - TODO: to_dict() is non-memory friendly - find alternate solution?
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(self.fasta_file, 'fasta'))

        # Iterate over DataFrame of BED windows
        for bed_window in self.bed_windows.itertuples():

            # Retrieve nucleotide sequence defined by BED coordinates
            if bed_window.chrom in fasta_sequences:  # TODO: will our FASTA file have sequences named by chromosome?
                nucleotide_sequence = np.array(fasta_sequences[bed_window.chrom].seq[bed_window.start:bed_window.end]) # TODO: do we care about strandedness of BED windows?

                # One-Hot encode nucleotide sequence
                nucleotide_sequence = nucleotide_sequence.reshape((-1, 1))
                onehot_sequence_matrix = encoder.fit_transform(nucleotide_sequence)
                onehot_sequences_matrices.append(onehot_sequence_matrix)

            # TODO: handle case where BED window sequence not found in FASTA file

        self.onehot_sequence_matrices = onehot_sequences_matrices

    def extract_transition_matrices(self):
        """
        Constructs transition matrices based on read alignments in BAM files
        for each BED window.

        Sets:
        -----
        self.transition_matrices : list of np.ndarray
            Transition matrices capturing state changes for reads per BED window.
        """

        transition_matrices = [] # TODO: convert to numpy array?

        # Iterate over DataFrame of BED windows
        for window in self.bed_windows.itertuples():

            matrix_shape = (12, window.end - window.start)
            transition_matrix = np.zeros(matrix_shape, dtype=int)

            # Iterate over BAM files
            for bam_file in self.bam_files: # TODO: stipulate that bam files are split by chromosome (for efficiency)?

                # Iterate over reads within BED window (even partially) - TODO: use pysam pileup instead?
                samfile = pysam.AlignmentFile(bam_file, 'rb')
                for read in samfile.fetch(window.chrom, window.start, window.end - 1): # BED window end coordinate is not inclusive, samfile region is inclusive, TODO: will our BAM file have reads named by chromosome?
                    self.cigar_transition_features(transition_matrix, window.start, window.end, read)

            transition_matrices.append(transition_matrix)

        # TODO: save these to disk instead
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
        cigar_tuples = [('S', 'M')] # TODO: better name for this
        for cigar_tuple in read.cigartuples:
            if cigar_tuple[0] == 0:
                cigar_tuples.extend([('M', cigar_tuple[1] - 1), ('M', 'N')])
            elif cigar_tuple[0] == 3:
                cigar_tuples.extend([('N', cigar_tuple[1] - 1), ('N', 'M')])
            else:
                return
        cigar_tuples[-1] = ('M', 'E')

        # Line up read to BED window
        read_offset = read.reference_start - window_start
        cigar_tuples, read_offset = self.trim_cigar_tuples(cigar_tuples, read_offset)

        # Add transition features as described by cigar_tuples
        reverse_offset = 6 if read.is_reverse else 0
        features_added = 0
        for cigar_tuple in cigar_tuples:

            # TODO: improve this code somewhat
            if not isinstance(cigar_tuple[1], int): # if non-self-loop transition (no repetition)
                transition_feat = self.transition_dict[(cigar_tuple[0], cigar_tuple[1])]
                if window_start + read_offset + features_added < window_end:
                    transition_matrix[transition_feat + reverse_offset][read_offset + features_added] += 1
                else:
                    return
                features_added += 1

            else: # self-loop condition (repetition)
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

        # "Fast-forward" the CIGAR tuple encoding of reads that begin before the BED window
        i = 0 # keep track of count of processed tuples, to be sliced out later
        # TODO: improve this code somewhat
        while read_offset < 0:
            repetition = cigar_tuples[i][1]
            if not isinstance(repetition, int): # if non-self-loop transition (no repetition)
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
        import os
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        for i, (seq_matrix, trans_matrix) in enumerate(zip(self.onehot_sequence_matrices, self.transition_matrices)):
            # Save one-hot sequence matrix
            seq_filename = os.path.join(self.output_dir, f'onehot_sequence_matrix_{i}.npy')
            np.save(seq_filename, seq_matrix)
            
            # Save transition matrix
            trans_filename = os.path.join(self.output_dir, f'transition_matrix_{i}.npy')
            np.save(trans_filename, trans_matrix)
            
            print(f"Saved matrices for window {i}:")
            print(f"  Sequence matrix: {seq_filename}")
            print(f"  Transition matrix: {trans_filename}")

def main():
    """Parse command-line arguments and run encoding process."""
    parser = argparse.ArgumentParser(description='One-hot encode genomic sequences and RNA-seq reads.')
    
    parser.add_argument('-b', '--bed', 
                        nargs='+', 
                        required=True, 
                        help='Path to one or more BED files')
    
    parser.add_argument('-f', '--fasta', 
                        required=True, 
                        help='Path to FASTA file containing genomic sequences')
    
    parser.add_argument('-a', '--bam', 
                        nargs='+', 
                        required=True, 
                        help='Path to one or more BAM files')
    
    parser.add_argument('-o', '--output', 
                        default='.', 
                        help='Output directory for saving matrices (default: current directory)')
    
    parser.add_argument('-p', '--print', 
                        action='store_true', 
                        help='Print matrices to console in addition to saving')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create Encoder instance
    encoder = Encoder(
        bed_files=args.bed, 
        fasta_file=args.fasta, 
        bam_files=args.bam,
        output_dir=args.output
    )
    
    # Run encoding process
    encoder.encode()
    
    # Save matrices
    encoder.save_matrices()
    
    # Optionally print matrices to console
    if args.print:
        for i, matrices in enumerate(zip(encoder.onehot_sequence_matrices, encoder.transition_matrices)):
            print(f'\nOne-Hot Sequence Matrix {i}:\n{matrices[0]}\n')
            print(f'\nTransition Matrix {i}:\n{matrices[1]}\n')

if __name__ == '__main__':
    main()
