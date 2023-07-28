import argparse, sys

def load_arg_dict():
    ################# use argparse to get CL args and make sure we are kosher
    parser = argparse.ArgumentParser(
        description="calculate feature vectors, train, or predict with diploSHIC"
    )
    parser._positionals.title = "possible modes (enter 'python diploSHIC.py modeName -h' for modeName's help message"
    subparsers = parser.add_subparsers(help="sub-command help")
    parser_c = subparsers.add_parser(
        "fvecSim", help="Generate feature vectors from simulated data"
    )
    parser_e = subparsers.add_parser(
        "makeTrainingSets",
        help="Combine feature vectors from muliple fvecSim runs into 5 balanced training sets",
    )
    parser_a = subparsers.add_parser("train", help="train and test a shic CNN")
    parser_d = subparsers.add_parser(
        "fvecVcf", help="Generate feature vectors from data in a VCF file"
    )
    parser_b = subparsers.add_parser(
        "predict", help="perform prediction using an already-trained SHIC CNN"
    )
    parser_a.add_argument("trainDir", help="path to training set files")
    parser_a.add_argument(
        "testDir", help="path to test set files, can be same as trainDir"
    )
    parser_a.add_argument(
        "outputModel",
        help="file name for output model, will create two files one with structure one with weights",
    )
    parser_a.add_argument(
        "--epochs",
        type=int,
        help="max epochs for training CNN (default = 100)",
        default=50,
    )
    parser_a.add_argument(
        "--domain-adaptation",
        action='store_true',
        help="Optional Flag to run model with Domain Adaptation",
        default=False,
    )
    parser_a.add_argument(
        "--da-weight",
        type=float,
        help="Relative weight of DA Discriminator loss: Predictor loss",
        default=1,
    )
    parser_a.add_argument(
        "--pred-weight",
        type=float,
        help="Relative weight of Predictor loss: DA Discriminator loss",
        default=1,
    )
    parser_a.add_argument(
        "--empiricalDir",
        type=str,
        help="Path to directory containing empirical data for domain adaptation",
        default = None,
    )   
    parser_a.add_argument(
        "--numSubWins",
        type=int,
        help="number of subwindows that our chromosome is divided into (default = 11)",
        default=11,
    )
    parser_a.add_argument(
        "--confusionFile",
        help="optional file to which confusion matrix plot will be written (default = None)",
        default=None,
    )
    parser_a.add_argument(
        "--lossAccFile",
        help="optional file to which the model training metrics plot will be written (default = None)",
        default=None,
    )
    parser_a.set_defaults(mode="train")
    parser_a._positionals.title = "required arguments"
    parser_b.add_argument(
        "modelStructure", help="path to CNN structure .json file"
    )
    parser_b.add_argument("modelWeights", help="path to CNN weights .h5 file")
    parser_b.add_argument("predictFile", help="input file to predict")
    parser_b.add_argument("predictFileOutput", help="output file name")
    parser_b.add_argument(
        "--numSubWins",
        type=int,
        help="number of subwindows that our chromosome is divided into (default = 11)",
        default=11,
    )
    parser_b.add_argument(
        "--simData",
        help="Are we using simulated input data wihout coordinates?",
        action="store_true",
    )
    parser_b.set_defaults(mode="predict")
    parser_b._positionals.title = "required arguments"

    parser_c.add_argument(
        "shicMode",
        help="specifies whether to use original haploid SHIC (use 'haploid') or diploSHIC ('diploid')",
        default="diploid",
    )
    parser_c.add_argument(
        "msOutFile",
        help="path to simulation output file (must be same format used by Hudson's ms)",
    )
    parser_c.add_argument(
        "fvecFileName", help="path to file where feature vectors will be written"
    )
    parser_c.add_argument(
        "--totalPhysLen",
        type=int,
        help="Length of simulated chromosome for converting infinite sites ms output to finite sites (default=1100000)",
        default=1100000,
    )
    parser_c.add_argument(
        "--numSubWins",
        type=int,
        help="The number of subwindows that our chromosome will be divided into (default=11)",
        default=11,
    )
    parser_c.add_argument(
        "--maskFileName",
        help=(
            "Path to a fasta-formatted file that contains masking information (marked by 'N'). "
            "If specified, simulations will be masked in a manner mirroring windows drawn from this file."
        ),
        default="None",
    )
    parser_c.add_argument(
        "--vcfForMaskFileName",
        help=(
            "Path to a VCF file that contains genotype information. This will be used to mask genotypes "
            "in a manner that mirrors how the true data are masked."
        ),
        default=None,
    )
    parser_c.add_argument(
        "--popForMask",
        help="The label of the population for which we should draw genotype information from the VCF for masking purposes.",
        default=None,
    )
    parser_c.add_argument(
        "--sampleToPopFileName",
        help=(
            "Path to tab delimited file with population assignments (used for genotype masking); format: "
            "SampleID\tpopID"
        ),
        default="None",
    )
    parser_c.add_argument(
        "--unmaskedGenoFracCutoff",
        type=float,
        help="Fraction of unmasked genotypes required to retain a site (default=0.75)",
        default=0.75,
    )
    parser_c.add_argument(
        "--chrArmsForMasking",
        help=(
            "A comma-separated list (no spaces) of chromosome arms from which we want to draw masking "
            "information (or 'all' if we want to use all arms. Ignored if maskFileName is not specified."
        ),
        default="all",
    )
    parser_c.add_argument(
        "--unmaskedFracCutoff",
        type=float,
        help="Minimum fraction of unmasked sites, if masking simulated data (default=0.25)",
        default=0.25,
    )
    parser_c.add_argument(
        "--outStatsDir",
        help="Path to a directory where values of each statistic in each subwindow are recorded for each rep",
        default="None",
    )
    parser_c.add_argument(
        "--ancFileName",
        help=(
            "Path to a fasta-formatted file that contains inferred ancestral states ('N' if unknown)."
            " This is used for masking, as sites that cannot be polarized are masked, and we mimic this in the simulted data."
            " Ignored in diploid mode which currently does not use ancestral state information"
        ),
        default="None",
    )
    parser_c.add_argument(
        "--pMisPol",
        type=float,
        help="The fraction of sites that will be intentionally polarized to better approximate real data (default=0.0)",
        default=0.0,
    )
    parser_c.set_defaults(mode="fvecSim")
    parser_c._positionals.title = "required arguments"

    parser_d.add_argument(
        "shicMode",
        help="specifies whether to use original haploid SHIC (use 'haploid') or diploSHIC ('diploid')",
    )
    parser_d.add_argument(
        "chrArmVcfFile",
        help="VCF format file containing data for our chromosome arm (other arms will be ignored)",
    )
    parser_d.add_argument(
        "chrArm",
        help="Exact name of the chromosome arm for which feature vectors will be calculated",
    )
    parser_d.add_argument("chrLen", type=int, help="Length of the chromosome arm")
    parser_d.add_argument(
        "fvecFileName", help="path to file where feature vectors will be written"
    )
    parser_d.add_argument(
        "--targetPop",
        help="Population ID of samples we wish to include",
        default="None",
    )
    parser_d.add_argument(
        "--sampleToPopFileName",
        help=(
            "Path to tab delimited file with population assignments; format: "
            "SampleID\tpopID"
        ),
        default="None",
    )
    parser_d.add_argument(
        "--winSize",
        type=int,
        help="Length of the large window (default=1100000)",
        default=1100000,
    )
    parser_d.add_argument(
        "--numSubWins",
        type=int,
        help="Number of sub-windows within each large window (default=11)",
        default=11,
    )
    parser_d.add_argument(
        "--maskFileName",
        help=(
            "Path to a fasta-formatted file that contains masking information (marked by 'N'); "
            "must have an entry with title matching chrArm"
        ),
        default="None",
    )
    parser_d.add_argument(
        "--unmaskedFracCutoff",
        type=float,
        help="Fraction of unmasked sites required to retain a subwindow (default=0.25)",
        default=0.25,
    )
    parser_d.add_argument(
        "--unmaskedGenoFracCutoff",
        type=float,
        help="Fraction of unmasked genotypes required to retain a site (default=0.75)",
        default=0.75,
    )
    parser_d.add_argument(
        "--ancFileName",
        help=(
            "Path to a fasta-formatted file that contains inferred ancestral states ('N' if unknown); "
            "must have an entry with title matching chrArm. Ignored for diploid mode which currently does not use ancestral "
            "state information."
        ),
        default="None",
    )
    parser_d.add_argument(
        "--statFileName",
        help="Path to a file where statistics will be written for each subwindow that is not filtered out",
        default="None",
    )
    parser_d.add_argument(
        "--segmentStart",
        help="Left boundary of region in which feature vectors are calculated (whole arm if omitted)",
        default="None",
    )
    parser_d.add_argument(
        "--segmentEnd",
        help="Right boundary of region in which feature vectors are calculated (whole arm if omitted)",
        default="None",
    )
    parser_d.set_defaults(mode="fvecVcf")
    parser_d._positionals.title = "required arguments"

    parser_e.add_argument(
        "neutTrainingFileName", help="Path to our neutral feature vectors"
    )
    parser_e.add_argument(
        "softTrainingFilePrefix",
        help=(
            "Prefix (including higher-level path) of files containing soft training examples"
            "; files must end with '_$i.$ext' where $i is the subwindow index of the sweep and $ext is any extension."
        ),
    )
    parser_e.add_argument(
        "hardTrainingFilePrefix",
        help=(
            "Prefix (including higher-level path) of files containing hard training examples"
            "; files must end with '_$i.$ext' where $i is the subwindow index of the sweep and $ext is any extension."
        ),
    )
    parser_e.add_argument(
        "sweepTrainingWindows",
        type=int,
        help=(
            "comma-separated list of windows to classify as sweeps (usually just '5'"
            " but without the quotes)"
        ),
    )
    parser_e.add_argument(
        "linkedTrainingWindows",
        help=(
            "list of windows to treat as linked to sweeps (usually '0,1,2,3,4,6,7,8,9,10' but"
            " without the quotes)"
        ),
    )
    parser_e.add_argument(
        "outDir", help="path to directory where the training sets will be written"
    )
    parser_e.set_defaults(mode="makeTrainingSets")
    parser_e._positionals.title = "required arguments"
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    argsDict = vars(args)
    return argsDict