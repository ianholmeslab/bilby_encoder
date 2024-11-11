"""Helper methods to generate and save "realistic" test data for bilby_encoder."""
# pylint: disable=fixme, disable=no-member

import os
import re
from itertools import groupby

import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
#from Bio.pairwise2 import format_alignment
import pysam

# TODO: generate_splice_structure()?
def generate_gene(splice_structure: list[int]) -> tuple[str, str]:
    """
    Generates a "realistic" gene sequence with intron and exon regions based on a specified 
    structure.

    This function simulates a gene sequence by generating nucleotide sequences for introns 
    and exons according to a given structure. Introns are flanked by canonical "GT...AG" splice 
    sites. Exons are included both in the full gene sequence and in a separate "read" sequence 
    containing only exons.

    Parameters:
    ----------
    splice_structure : list of int
        A list of integers representing the lengths of consecutive intron and exon segments.
        The list should have an odd number of elements, starting and ending with exon lengths.
        Exon lengths are given by elements at even indices (0, 2, 4, ...),
        and intron lengths by elements at odd indices (1, 3, 5, ...).

    Returns:
    -------
    tuple of (str, str)
        - gene_sequence: A string representing the nucleotide sequence of the gene, 
          including introns and exons.
        - exon_sequence: A string representing the nucleotide sequence of the exons 
          only (the "read").

    Raises:
    ------
    ValueError
        If the length of `splice_structure` is not odd, if any exon length (elements at even 
        indices) is less than 1 nucleotide, or if any intron length (elements at odd indices)
        is less than 4 nucleotides.

    Examples:
    --------
    >>> generate_gene([5, 6, 3])
    ('ATCGACTGCCAGAT', 'ATCGA')  # Example output, values will vary

    Notes:
    -----
    - Each intron is generated with a minimum length of 4 nucleotides, and is flanked by "GT" 
      and "AG" to represent canonical splice donor and acceptor sites.
    - Exon sequences are directly appended to the "exon_sequence", while both introns and exons 
      are added to the full "gene_sequence".
    - The output is randomized, so nucleotide sequences will differ on each function call.
    """

    if len(splice_structure) % 2 != 1:
        raise ValueError("The splice structure must contain an odd number of"
                         "segments, alternating between exons and introns.")

    if any(splice_structure[i] < 1 for i in range(0, len(splice_structure), 2)):
        raise ValueError("Exons must be at least 1 nucleotide long.")

    if any(splice_structure[i] < 4 for i in range(1, len(splice_structure), 2)):
        raise ValueError("Introns must be at least 4 nucleotides long.")

    nucleotides = np.array(['A', 'T', 'C', 'G'])
    gene_segments, exon_segments = [], []

    is_exon = True
    for splice_size in splice_structure:
        if is_exon:
            # Generate exon sequence
            exon_seq = ''.join(np.random.choice(nucleotides, splice_size))
            exon_segments.append(exon_seq)
            gene_segments.append(exon_seq)
        else:
            # Generate intron sequence with canonical splice sites
            intron_seq = 'GT' + ''.join(np.random.choice(nucleotides, splice_size - 4)) + 'AG'
            gene_segments.append(intron_seq)

        is_exon = not is_exon

    # Join segments into final sequences
    gene_sequence = ''.join(gene_segments)
    exon_sequence = ''.join(exon_segments)

    return gene_sequence, exon_sequence

def random_non_negative_int_array_sum_to_n(n: int, size: int) -> np.ndarray[int]:
    """
    Generates an array of non-negative integers that sum to a specified target value.

    This function divides the target integer `n` into `size` non-negative integer values 
    that add up to `n`. The distribution of values is random and non-negative, providing
    a way to split a total sum randomly across a specified number of elements.

    Parameters:
    ----------
    n : int
        The target sum for the array. Must be a non-negative integer.
    size : int
        The number of elements in the output array. Must be at least 1.

    Returns:
    -------
    numpy.ndarray
        An array of length `size` containing non-negative integers that sum to `n`.

    Raises:
    ------
    ValueError
        If `n` is negative or if `size` is less than 1.

    Examples:
    --------
    >>> random_non_negative_int_array_sum_to_n(10, 3)
    array([3, 2, 5])  # Example output, values will vary

    Notes:
    -----
    - If `size` is 1, the array will contain only `[n]`.
    - The output is randomized, so values will differ on each call.
    """

    if n < 0:
        raise ValueError("The target sum `n` must be a non-negative integer.")
    if size < 1:
        raise ValueError("The array size must be at least 1.")

    if size == 1:
        return np.array([n])  # Only one element, which must be the sum itself

    # Generate `size - 1` random cut points in the range [0, n]
    cut_points = np.sort(np.random.randint(0, n + 1, size - 1))

    # Calculate differences between consecutive cut points to get the parts
    values = np.diff(np.concatenate(([0], cut_points, [n])))

    return values

def generate_genome(genes: list[str], size: int) -> str:
    """
    Generates a "realistic" genome sequence with genes interspersed with random-length 
    non-coding regions, adjusted to a specified total genome size.

    This function creates a genome by arranging given gene sequences with randomly sized 
    non-coding regions between them. The resulting genome is padded as needed with additional 
    non-coding sequence to achieve the exact specified size.

    Parameters:
    ----------
    genes : list of str
        List of gene sequences (strings) to be included in the genome. Each gene is assumed 
        to be a nucleotide sequence.
    size : int
        The total length of the genome to be generated. Must be at least as long as the 
        total length of the gene sequences.

    Returns:
    -------
    str
        A nucleotide sequence representing the genome, with genes separated and padded 
        by non-coding sequences to reach the specified size.

    Raises:
    ------
    ValueError
        If the combined length of genes exceeds `size`, or if no genes are provided.

    Examples:
    --------
    >>> generate_genome(["ATGCG", "TTAAC"], 20)
    'AATTATGCGGGTTAACCCGA'  # Example output, values will vary

    Notes:
    -----
    - Each non-coding region is randomly generated as a sequence of 'A', 'T', 'C', or 'G'.
    - The function ensures that the final genome length is exactly `size`.
    """

    num_genes = len(genes)
    if num_genes == 0:
        raise ValueError("At least one gene must be provided.")

    genes_length = sum(len(gene) for gene in genes)
    if genes_length > size:
        raise ValueError("Total length of genes cannot exceed the specified genome size.")

    # Calculate the total length needed for non-coding regions
    non_coding_length = size - genes_length

    # Generate random non-coding region lengths that sum to `non_coding_length`
    non_coding_regions = random_non_negative_int_array_sum_to_n(non_coding_length, num_genes + 1)

    nucleotides = np.array(['A', 'T', 'C', 'G'])
    genome_segments = []

    # Construct the genome by alternating non-coding regions and genes
    for i in range(num_genes):
        genome_segments.append(''.join(np.random.choice(nucleotides, non_coding_regions[i])))
        genome_segments.append(genes[i])
    # Add the final non-coding region
    genome_segments.append(''.join(np.random.choice(nucleotides, non_coding_regions[-1])))

    # Join all parts into the final genome sequence
    genome_sequence = ''.join(genome_segments)

    return genome_sequence

def random_contiguous_substring(string: str, length: int) -> str:
    """
    Generates a random contiguous substring of a specified length from the given string.

    This function selects a random starting position in the input string, ensuring that
    the selected substring has the desired length. It returns a contiguous substring 
    of the specified length, which is drawn from the given string.

    Parameters:
    ----------
    string : str
        The original string from which the substring will be drawn.
    length : int
        The length of the desired substring. Must be less than or equal to the length of `string`.

    Returns:
    -------
    str
        A random contiguous substring of the given string with the specified length.

    Raises:
    ------
    ValueError
        If `length` is greater than the length of `string`.

    Examples:
    --------
    >>> random_contiguous_substring("abcdef", 3)
    'bcd'  # Example output, values will vary
    >>> random_contiguous_substring("ATCGTAGC", 4)
    'CGTA'  # Example output, values will vary

    Notes:
    -----
    - If the `length` is equal to the length of the string, the function will return 
      the entire string.
    - The returned substring is drawn randomly, so it will differ on each function call.
    """

    if length > len(string):
        raise ValueError("The desired substring length cannot exceed the"
                         "length of the original string.")

    # Randomly select a starting index within bounds that allow for a substring of the given length
    start_idx = np.random.randint(0, len(string) - length + 1)
    return string[start_idx:start_idx + length]

def generate_reads(exon_sequences: list[str], read_counts: list[int]) -> list[str]:
    """
    Generate random reads from the exon sequences of genes.

    This function generates random subsequences (reads) from each of the provided exon sequences. 
    The number of each reads generated for each exon sequence is defined by the corresponding entry 
    in the `read_counts` list. Each read is randomly is extracted from the respective exon sequence.

    Parameters:
    ----------
    exon_sequences : list of str
        A list of exon sequences (strings), each corresponding to a different gene.
    read_counts : list of int
        A list of integers specifying how many reads to generate from each exon sequence. 
        The length of this list must match `exon_sequences`.

    Returns:
    -------
    list of str
        A list of randomly selected reads, each extracted from each of the exon sequences.

    Raises:
    ------
    ValueError
        If the length of `exon_sequences` does not match `read_counts`, or if any `read_count`
        exceeds the length of the corresponding exon sequence.

    Examples:
    --------
    >>> generate_reads(["ATGCGCTGTCACTAT", "TTAACGTACCTAGCG"], [3, 2])
    ['GCTGTCAC', 'CACTAT', 'ATCGCTGTCAC', 'CGTACCT', 'AACGTACCTAGCG']  # Example output, values will vary

    Notes:
    -----
    - Each read is a random contiguous substring of the corresponding exon sequence.
    - The read length is randomly chosen, but must not exceed the length of the exon sequence.
    """

    if len(exon_sequences) != len(read_counts):
        raise ValueError("The length of `exon_sequences` must match the length of `read_counts`.")

    # Generate random reads
    reads = []
    for exon_seq, count in zip(exon_sequences, read_counts):

        # Ensure valid read counts
        if count < 1:
            raise ValueError("Each `read_count` must be greater than zero.")

        # Generate the requested number of reads for this exon sequence
        for _ in range(count):
            read_len = np.random.randint(5, len(exon_seq) + 1)  # Random read length
            read = random_contiguous_substring(exon_seq, read_len)
            reads.append(read)

    return reads

def partition_reads(reads: list[str], num_partitions: int) -> list[list[str]]:
    """
    Randomly partitions a list of reads into a specified number of sublists.

    Parameters:
    ----------
    reads : list of str
        The original list of reads to partition.
    num_partitions : int
        The number of partitions (sublists) to create.

    Returns:
    -------
    list of list of str
        A list containing `num_partitions` sublists, each with a random subset of the reads.
    """

    # Shuffle the list of reads to ensure random distribution
    np.random.shuffle(reads)

    # Split the shuffled reads list into approximately equal sublists
    partitions = np.array_split(reads, num_partitions)

    # Convert numpy arrays to lists and return
    return [list(partition) for partition in partitions]

def save_test_case(data_dir: str, test_case_name: str, genome: str, reads: list[list[str]],
                   windows: list[list[tuple[str, int, int]]]) -> None:
    """
    Save test case data including genome, reads, and windows in FASTA, BAM, and BED formats.

    Parameters:
    - data_dir (str): Base directory for storing test cases.
    - test_case_name (str): Name of the test case, used as a subdirectory within data_dir.
    - genome (str): Reference genome sequence to save as a FASTA file.
    - reads (list[list[str]]): Nested list of read sequences; each inner list represents 
      reads for a specific file.
    - windows (list[list[tuple[str, int, int]]]): Nested list of genomic intervals 
      as (chrom, start, end) tuples for each file.

    The function generates:
    - genome.fasta: Reference genome in FASTA format.
    - reads{index}.fasta: FASTA files with reads for each file in `reads`.
    - alignment{index}.bam: BAM files with aligned reads for each file in `reads`.
    - windows{index}.bed: BED files specifying genomic windows for each list in `windows`.

    Notes:
    - reads{index}.fasta is not used by the Encoder() class. This file is just used for manual inspection of test data.
    """

    # Make directory to save test case files
    testcase_directory = os.path.join(data_dir, test_case_name)
    os.makedirs(testcase_directory, exist_ok=True)
    print(f"Directory '{testcase_directory}' created.")

    # Write genome to FASTA file
    fasta_path = os.path.join(testcase_directory, "genome.fasta")
    with open(fasta_path, "w", encoding="utf-8") as output_handle:
        record = SeqRecord(Seq(genome), id="chr1")
        SeqIO.write(record, output_handle, "fasta")

    # Write reads to separate FASTA files for each group of reads
    for i, file_reads in enumerate(reads):
        fasta_path = os.path.join(testcase_directory, f"reads{i}.fasta")
        with open(fasta_path, "w", encoding="utf-8") as output_handle:
            for j, read in enumerate(file_reads):
                record = SeqRecord(Seq(read), id=f"read{j}")
                SeqIO.write(record, output_handle, "fasta")

    # Write BAM files with aligned reads for each set of reads
    # TODO: this is super hack-y - there has to be a better way to do small alignments
    bam_contents = generate_bam_file_contents(reads, genome)
    bam_header = { 'HD': {'VN': '1.0', 'SO':'unsorted'},
                   'SQ': [{'LN': len(genome), 'SN': 'chr1'}] }

    for i, bam_file_contents in enumerate(bam_contents):
        bam_path = os.path.join(testcase_directory, f"alignment{i}.bam")
        with pysam.AlignmentFile(bam_path, "wb", header=bam_header) as outf:
            for j, bam_row in enumerate(bam_file_contents):
                a = pysam.AlignedSegment()
                a.query_name = f"read{j}"
                a.query_sequence = reads[i][j]
                a.flag = 0
                a.reference_id = 0
                a.reference_start = bam_row[0]
                a.mapping_quality = 255
                a.cigar = bam_row[1]
                outf.write(a)

    # Write BED files specifying genomic windows
    for i, file_windows in enumerate(windows):
        bed_path = os.path.join(testcase_directory, f"windows{i}.bed")
        bed_df = pd.DataFrame(file_windows, columns=['chrom', 'start', 'end'])
        bed_df.to_csv(bed_path, sep='\t', index=False)

def generate_bam_file_contents(reads: list[list[str]], genome: str) -> list[list[tuple[int, list[tuple[int, int]]]]]:
    """
    Generate BAM file contents by aligning reads to the reference genome.

    Parameters:
    - reads (list[list[str]]): Nested list of read sequences for each file.
    - genome (str): Reference genome sequence.

    Returns:
    - List of alignment information for each file's reads, containing tuples of start position and CIGAR operations.
    """
    contents = []

    # Perform local alignments for each query
    for file_reads in reads:

        file_contents = []

        for read in file_reads:
            alignments = pairwise2.align.localms(genome, read, 2, -1, -0.5, -0.1)
            # Arbitrarily choose an alignment for each read (these alignments all have the same score)
            alignment = alignments[np.random.randint(0, len(alignments))]
            alignment_str = process_alignment_sequence(alignment)
            alignment_cigar = run_length_encoding(alignment_str)
            file_contents.append((alignment.start, alignment_cigar))

        contents.append(file_contents)

    return contents

def process_alignment_sequence(alignment) -> str:
    """
    Process alignment sequence by stripping unaligned nucleotides and converting 
    to CIGAR-like symbols.

    Parameters:
    - alignment (pairwise2.Alignment): Pairwise alignment object.

    Returns:
    - str: Sequence with 'M' for matches and 'N' for gaps.
    """
    aligned_seq = alignment.seqB.strip('-')
    return re.sub(r'[ATCG]', 'M', aligned_seq).replace('-', 'N')

def run_length_encoding(data: str) -> list[tuple[int, int]]:
    """
    Encode a sequence using run-length encoding for CIGAR representation.

    Parameters:
    - data (str): Sequence of 'M' and 'N' symbols.

    Returns:
    - list[tuple[int, int]]: List of (CIGAR operation, run length).
    """
    cigar_code_dict = {'M': 0, 'N': 3}
    return [(cigar_code_dict[char], sum(1 for _ in group)) for char, group in groupby(data)]
