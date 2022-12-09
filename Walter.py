#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import os
import re
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view
from operator import itemgetter
from tqdm import tqdm
import csv
import subprocess as sp

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_args():
	"""
	Get arguments from command line with argparse.
	"""
	
	parser = argparse.ArgumentParser(
		prog='Walter.py',
		description="""Calculate CpG positions and scores from an aligned bam file. Outputs raw and 
		coverage-filtered results in bed and bigwig format, including haplotype-specific results (when available).""")
	parser.add_argument("-b", "--bam",
						required=True,
						#default="test.bam",
						metavar="input.bam",
						help="The aligned BAM file. This file must be sorted and indexed.")
	parser.add_argument("-f", "--fasta",
						required=True,
						#default="test.fa",
						metavar="ref.fasta",
						help="The reference fasta file.")
	parser.add_argument("-o", "--output_label",
						required=True,
						#default="test",
						metavar="label",
						help="Label for output files, which results in [label].bed/bw.")
	parser.add_argument("-m", "--modsites",
						required=False,
						choices=["denovo", "reference"],
						default="denovo",
						help="Only output CG sites with a modification probability > 0 "
							 "(denovo), or output all CG sites based on the "
							 "supplied reference fasta (reference). [default = %(default)s]")
	parser.add_argument("-p", "--pileup_mode",
						required=False,
						choices=["model", "count"],
						default="model",
						help="Use a model-based approach to score modifications across sites (model) "
							 "or a simple count-based approach (count). [default = %(default)s]")
	parser.add_argument("-d", "--model_dir",
						required=False,
						default=None,
						metavar="/path/to/model/dir",
						help="Full path to the directory containing the model (*.pb files) to load. [default = None]")
	parser.add_argument("-c", "--min_coverage",
						required=False,
						default=4,
						type=int,
						metavar="int",
						help="Minimum coverage required for filtered outputs. [default: %(default)d]")
	parser.add_argument("-q", "--min_mapq",
						required=False,
						default=0,
						type=int,
						metavar="int",
						help="Ignore alignments with MAPQ < N. [default: %(default)d]")
	parser.add_argument("-a", "--hap_tag",
						required=False,
						default="HP",
						metavar="TAG",
						help="The SAM tag containing haplotype information. [default: %(default)s]")
	parser.add_argument("-s", "--chunksize",
						required=False,
						default=500000,
						type=int,
						metavar="int",
						help="Break reference regions into chunks "
							 "of this size for parallel processing. [default = %(default)d]")
	parser.add_argument("-t", "--threads",
						required=False,
						default=4,
						type=int,
						metavar="int",
						help="Number of threads for parallel processing. [default = %(default)d]")
	
	return parser.parse_args()


def validate_args(args):
	
	def error_exit(msg):
		raise Exception(msg)
	
	def check_required_file(file, label):
		if not os.path.isfile(file):
			error_exit(f"Can't find {label} file '{file}'")
	
	check_required_file(args.bam, "input bam")
	
	def is_bam_index_found(bam_file):
		bam_index_extensions = (".bai", ".csi")
		for ext in bam_index_extensions:
			bam_index_file=bam_file+ext
			if os.path.isfile(bam_index_file):
				return True
		return False
	
	if not is_bam_index_found(args.bam):
		error_exit(f"Can't find index for bam file '{args.bam}'")
	
	check_required_file(args.fasta, "reference fasta")
	
	if args.pileup_mode == "model":
		if args.model_dir is None:
			error_exit("Must supply a model to use when running model-based scoring")
		else:
			if not os.path.isdir(args.model_dir):
				error_exit("{} is not a valid directory path!".format(args.model_dir))
	else:
		if args.model_dir is not None:
			error_exit("Model directory is not used unless model-based scoring is selected")


def setup_logging(output_label):
	"""
	Set up logging to file.
	"""
	logname = "{}.Walter.log".format(output_label)
	# ensure logging file does not exist, if so remove
	if os.path.exists(logname):
		os.remove(logname)
	
	# set up logging to file
	logging.basicConfig(filename=logname,
						format="%(asctime)s: %(levelname)s: %(message)s",
						datefmt='%d-%b-%y %H:%M:%S',
						level=logging.DEBUG)


def log_args(args):
	"""
	Record argument settings in log file.
	"""
	logging.info("Using following argument settings:")
	for arg, val in vars(args).items():
		logging.info("\t--{}: {}".format(arg, val))


def get_regions_to_process(input_bam, input_fasta, chunksize, modsites, pileup_mode, model_dir, min_mapq, hap_tag):
	"""
	Breaks reference regions into smaller regions based on chunk
	size specified. Returns a list of lists that can be used for
	multiprocessing. Each sublist contains:
	[bam path (str), fasta path (str), modsites (str),
	reference name (str), start coordinate (int), stop coordinate (int)]
	
	:param input_bam: Path to input bam file. (str)
	:param input_fasta: Path to reference fasta file. (str)
	:param chunksize: Chunk size (default = 500000). (int)
	:param modsites: Filtering method. (str: "denovo", "reference")
	:param pileup_mode: Site modification calling method. (str: "model", "count")
	:param model_dir: Full path to model directory to load (if supplied), otherwise is None.
	:param min_mapq: Minimum mapping quality score. (int)
	:param hap_tag: The SAM tag label containing haplotype information. (str)
	:return regions_to_process: List of lists containing region sizes. (list)
	"""
	logging.info("get_regions_to_process: Starting chunking.")
	# open the input bam file with pysam
	bamIn = pysam.AlignmentFile(input_bam, 'rb')
	# empty list to store sublists with region information
	regions_to_process = []
	# iterate over reference names and their corresponding lengths
	references = zip(bamIn.references, bamIn.lengths)
	for ref, length in references:
		start = 1
		while start < length:
			end = start + chunksize
			if end < length:
				regions_to_process.append(
					[input_bam, input_fasta, modsites, pileup_mode, model_dir, ref, start, end - 1, min_mapq, hap_tag])
			else:
				regions_to_process.append(
					[input_bam, input_fasta, modsites, pileup_mode, model_dir, ref, start, length, min_mapq, hap_tag])
			start = start + chunksize
	# close bam
	bamIn.close()
	logging.info("get_regions_to_process: Created {:,} region chunks.\n".format(len(regions_to_process)))
	
	return regions_to_process

def write_output_bed(label, modsites, min_coverage, bed_results):
	"""
	Writes output bed file(s) based on information in bed_merge_results (default).
	Separates results into total, hap1, and hap2. If haplotypes not available,
	only total is produced.
	
	The bed_merge_results list will contain slighty different information depending on the pileup_mode option,
	but the first 7 fields will be identical:
	
	count-based list
		[(0) ref name, (1) start coord, (2) stop coord, (3) mod probability, (4) haplotype, (5) coverage,
		(6) mod sites, (7) unmod sites, (8) mod score, (9) unmod score]
	
	OR
	model-based list
		[(0) ref name, (1) start coord, (2) stop coord, (3) mod probability, (4) haplotype, (5) coverage,
		(6) mod sites, (7) unmod sites, (8) adjusted probability]
	
	:param outname: Name of output bed file to write. (str)
	:param modsites: "reference" or "denovo", for the CpG detection mode. (str)
	:param min_coverage: Minimum coverage to retain a site. (int)
	:param bed_results: List of sublists with information to write the output bed file. (list)
	:return output_files: List of output bed file names that were successfully written. (list)
	"""
	logging.info("write_output_bed: Writing unfiltered output bed files.\n")
	out_total = "{}.Walter.combined.{}.bed".format(label, modsites)
	out_hap1 = "{}.Walter.hap1.{}.bed".format(label, modsites)
	out_hap2 = "{}.Walter.hap2.{}.bed".format(label, modsites)
	cov_total = "{}.Walter.combined.{}.mincov{}.bed".format(label, modsites, min_coverage)
	cov_hap1 = "{}.Walter.hap1.{}.mincov{}.bed".format(label, modsites, min_coverage)
	cov_hap2 = "{}.Walter.hap2.{}.mincov{}.bed".format(label, modsites, min_coverage)
	
	# remove any previous version of output files
	for f in [out_total, out_hap1, out_hap2, cov_total, cov_hap1, cov_hap2]:
		if os.path.exists(f):
			os.remove(f)
	
	with open(out_total, 'a') as fh_total:
		with open(out_hap1, 'a') as fh_hap1:
			with open(out_hap2, 'a') as fh_hap2:
				for i in bed_results:
					if i[4] == "Total":
						fh_total.write("{}\n".format("\t".join([str(j) for j in i])))
					elif i[4] == "hap1":
						fh_hap1.write("{}\n".format("\t".join([str(j) for j in i])))
					elif i[4] == "hap2":
						fh_hap2.write("{}\n".format("\t".join([str(j) for j in i])))
	
	# write coverage-filtered versions of bed files
	logging.info(
		"write_output_bed: Writing coverage-filtered output bed files, using min coverage = {}.\n".format(min_coverage))
	output_files = []
	for inBed, covBed in [(out_total, cov_total), (out_hap1, cov_hap1), (out_hap2, cov_hap2)]:
		# if haplotypes not present, the bed files are empty, remove and do not write cov-filtered version
		if os.stat(inBed).st_size == 0:
			os.remove(inBed)
		else:
			output_files.append(inBed)
			# write coverage filtered bed file
			with open(inBed, 'r') as fh_in, open(covBed, 'a') as fh_out:
				for line in fh_in:
					if int(line.split('\t')[5]) >= min_coverage:
						fh_out.write(line)
			# check to ensure some sites were written, otherwise remove
			if os.stat(covBed).st_size == 0:
				os.remove(covBed)
			else:
				output_files.append(covBed)
	
	return output_files

def make_bed_df(bed, pileup_mode):
	"""
	Construct a pandas dataframe from a bed file.
	
	count-based list
		[(0) ref name, (1) start coord, (2) stop coord, (3) % mod sites, (4) haplotype, (5) coverage,
		(6) mod sites, (7) unmod sites, (8) mod score, (9) unmod score]
	
	OR
	model-based list
		[(0) ref name, (1) start coord, (2) stop coord, (3) mod probability, (4) haplotype, (5) coverage,
		(6) mod sites, (7) unmod sites, (8) adjusted probability]
	
	:param bed: Name of bed file.
	:param pileup_mode: Site modification calling method. (str: "model", "count")
	:return df: Pandas dataframe.
	"""
	logging.debug("make_bed_df: Converting '{}' to pandas dataframe.\n".format(bed))
	if pileup_mode == "count":
		df = pd.read_csv(bed, sep='\t', header=None,
						 names=['chromosome', 'start', 'stop', 'mod_probability', 'haplotype', 'coverage',
								'modified_bases', 'unmodified_bases', 'mod_score', 'unmod_score'])
		df.drop(columns=['modified_bases', 'unmodified_bases', 'mod_score', 'unmod_score', 'haplotype', 'coverage'],
				inplace=True)
	
	elif pileup_mode == "model":
		df = pd.read_csv(bed, sep='\t', header=None,
						 names=['chromosome', 'start', 'stop', 'mod_probability', 'haplotype', 'coverage',
								'modified_bases', 'unmodified_bases', 'adj_prob'])
		df.drop(columns=['haplotype', 'coverage', 'modified_bases', 'unmodified_bases', 'adj_prob'], inplace=True)
	
	# df.sort_values(by=['chromosome', 'start'], inplace=True)
	
	return df


def get_bigwig_header_info(input_fasta):
	"""
	Get chromosome names and lengths from reference fasta.
	
	:param input_fasta: Name of reference fasta file.
	:return header: List of tuples, containing [ (ref1, length1), (ref2, length2), ...] .
	"""
	logging.debug("get_bigwig_header_info: Getting ref:length info from reference fasta.\n")
	header = []
	with open(input_fasta) as fh:
		for record in SeqIO.parse(fh, "fasta"):
			header.append((record.id, len(record.seq)))
	return header


def write_bigwig_from_df(df, header, outname):
	"""
	Function to write a bigwig file using a pandas dataframe from a bed file.
	
	:param df: Pandas dataframe object (created from bed file).
	:param header: List containing (ref name, length) information. (list of tuples)
	:param outname: Name of bigwig output file to write (OUT.bw).
	"""
	logging.debug("write_bigwig_from_df: Writing bigwig file for '{}'.\n".format(outname))
	# first filter reference contigs to match those in bed file
	# get all unique ref contig names from bed
	chroms_present = list(df["chromosome"].unique())
	# header is a list of tuples, filter to keep only those present in bed
	# must also sort reference contigs by name
	filtered_header = sorted([x for x in header if x[0] in chroms_present], key=itemgetter(0))
	for i, j in filtered_header:
		logging.debug("\tHeader includes: '{}', '{}'.".format(i, j))
	# raise error if no reference contig names match
	if not filtered_header:
		logging.error("No reference contig names match between bed file and reference fasta!")
		raise ValueError("No reference contig names match between bed file and reference fasta!")
	
	# open bigwig object, enable writing mode (default is read only)
	bw = pyBigWig.open(outname, "w")
	# must add header to bigwig prior to writing entries
	bw.addHeader(filtered_header)
	# iterate over ref contig names
	for chrom, length in filtered_header:
		logging.debug("\tAdding entries for '{}'.".format(chrom))
		# subset dataframe by chromosome name
		temp_df = df[df["chromosome"] == chrom]
		logging.debug("\tNumber of entries = {:,}.".format(temp_df.shape[0]))
		# add entries in order specified for bigwig objects:
		# list of chr names: ["chr1", "chr1", "chr1"]
		# list of start coords: [1, 100, 125]
		# list of stop coords: ends=[6, 120, 126]
		# list of vals: values=[0.0, 1.0, 200.0]
		bw.addEntries(list(temp_df["chromosome"]),
					  list(temp_df["start"]),
					  ends=list(temp_df["stop"]),
					  values=list(temp_df["mod_probability"]))
		logging.debug("\tFinished entries for '{}'.\n".format(chrom))
	# close bigwig object
	bw.close()


def convert_bed_to_bigwig(bed_files, fasta, pileup_mode):
	"""
	Write bigwig files for each output bed file.
	
	:param bed_files: List of output bed file names. (list)
	:param fasta: A path to reference fasta file. (str)
	:param pileup_mode: Site modification calling method. (str: "model", "count")
	"""
	logging.info("convert_bed_to_bigwig: Converting {} bed files to bigwig files.\n".format(len(bed_files)))
	header = get_bigwig_header_info(fasta)
	for bed in bed_files:
		outname = "{}.Walter.bw".format(bed.split(".bed")[0])
		df = make_bed_df(bed, pileup_mode)
		write_bigwig_from_df(df, header, outname)


def main():
	args = get_args()
	setup_logging(args.output_label)
	validate_args(args)
	log_args(args)
	
	print("\nChunking regions for multiprocessing.")
	regions_to_process = get_regions_to_process(args.bam, args.fasta, args.chunksize, args.modsites,
												args.pileup_mode, args.model_dir, args.min_mapq, args.hap_tag)
	with open("{}.Walter.regions".format(args.output_label), "w", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		writer.writerows(regions_to_process)
	
	#bed_results = run_all_pileup_processing(regions_to_process, args.threads)
	print("Running multiprocessing on {:,} chunks.".format(len(regions_to_process)))
	sp.call("mkdir Heisenberg_tmp".format(args.output_label), shell=True)
	sp.call("parallel --will-cite -j " + str(args.threads) + " -a " + args.output_label + ".Walter.regions -k --colsep '\t' --bar " + 
		"\"Heisenberg.py -b {1} -f {2} -m {3} -p {4} -d {5} -r {6} -s {7} -e {8} -q {9} -a {10} -o " + args.output_label + "\"", shell=True)
	
	print("Concatenating multiprocessing results")
	sp.call("parallel --will-cite -j 1 -a " + args.output_label + ".Walter.regions -k --colsep '\t' --bar " + 
		"\"" +
		"cat Heisenberg_tmp/{6}.{7}-{8}.bed >> " + args.output_label + ".Heisenberg.bed; " + 
		"cat Heisenberg_tmp/{6}.{7}-{8}.reads >> " + args.output_label + ".Heisenberg.reads; " +
		#"cat Heisenberg_tmp/{6}.{7}-{8}.NMV.tsv >> " + args.output_label + ".Heisenberg.NMV.tsv; " + 
		"cat Heisenberg_tmp/{6}.{7}-{8}.log >> " + args.output_label + ".Heisenberg.log\"", shell=True)
	sp.call("rm -r Heisenberg_tmp", shell=True)
	sp.call("gzip " + args.output_label + ".Heisenberg.reads", shell=True)
	
	print("Finished multiprocessing.\nWriting bed files.")
	with open(args.output_label + ".Heisenberg.bed", newline='') as f:
		reader = csv.reader(f, delimiter="\t")
		bed_results = list(reader)
	
	bed_files = write_output_bed(args.output_label, args.modsites, args.min_coverage, bed_results)
	
	print("Writing bigwig files.")
	convert_bed_to_bigwig(bed_files, args.fasta, args.pileup_mode)
	
	#print("Preparing per-read results for NanoMethViz & Differential Methylation")
	#sp.call("bgzip " + args.output_label + ".Heisenberg.NMV.tsv", shell=True)
	#sp.call("tabix -s 2 -b 3 -e 3 " + args.output_label + ".NMV.tsv.gz", shell=True)
	
	print("Finished.\n")


if __name__ == '__main__':
	main()

