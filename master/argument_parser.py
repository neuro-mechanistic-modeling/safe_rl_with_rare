'''
Argument parser for all algorithms.
'''
from racetrackgym.argument_parser import Racetrack_parser


class RAREParser(Racetrack_parser):
    def __init__(self):
        super().__init__()
        evaluation_stage = self.add_argument_group(
            "evaluation_stage", parent_groups=["hermes", "common"]
        )
        evaluation_stage.add_argument(
            "-esl",
            "--es_length_stage",
            help="Number of episodes between two evaluation stages",
            default=10000,
            type=int,
        )
        evaluation_stage.add_argument(
            "-esp",
            "--es_pre_training",
            help="Number of episodes before the first evaluation stage",
            default=10000,
            type=int,
        )
        evaluation_stage.add_argument(
            "-eseps",
            "--es_epsilon",
            help="DSMC epsilon parameter",
            default=0.05,
            type=float,
        )
        evaluation_stage.add_argument(
            "-eskap",
            "--es_kappa",
            help="DSMC kappa parameter",
            default=0.05,
            type=float,
        )
        evaluation_stage.add_argument(
            "-esir",
            "--es_initial_runs",
            help="DSMC number of initial runs",
            default=100,
            type=int,
        )
        evaluation_stage.add_argument(
            "-est",
            "--es_num_threads",
            help="number of threads used by evaluation",
            default=8,
            type=int,
        )
        evaluation_stage.add_argument(
            "-esch",
            "--es_use_chernoff_hoeffding",
            help="use ch method for evaluation",
            default=False,
            action="store_true",
        )
        evaluation_stage.add_argument(
            "-esmp",
            "--es_minimal_prio",
            help="minimal priority",
            default=0.2,
            type=float,
        )
        evaluation_stage.add_argument(
            "-esalpha", "--es_alpha", help="ES parameter alpha", default=1, type=float
        )
        evaluation_stage.add_argument(
            "-esg",
            "--es_grp",
            help="use grp for evaluation",
            default=False,
            action="store_true",
        )

        evaluation_stage.add_argument(
            "-exp",
            "--experiment_path",
            help="the root directory where hermes stores the experiments",
            default=None,
        )

        pr = self.add_argument_group("Prioritized Replay")
        pr.add_argument("--pr_alpha", type=float, default=0.5)
        pr.add_argument("--pr_beta", type=float, default=0.5)
        pr.add_argument("--pr_min_prio", type=float, default=0.05)

        minigrid = self.add_argument_group("minigrid")
        minigrid.add_argument("--num_obstacles",
                                   help="Number of obstacles",
                                   type=int,
                                   default=6)

        rare = self.add_argument_group("rare")
        rare.add_argument("--benchmark",
                               help="Racetrack or MiniGrid?",
                               type=str)
        rare.add_argument("--relevance_heuristic",
                               help="Evaluate states' relevance using value or novelty heuristic?",
                               type=str,
                               default="value")
        rare.add_argument("--reduction_strategy",
                               help="Reduce archives using cluster or maximal distance strategy?",
                               type=str,
                               default="cluster")
        rare.add_argument("--archive_size_factor",
                               help="Determines sizes of archive relative of number of free cells in environment",
                               type=float,
                               default=0.25)
        rare.add_argument("--psi_min",
                               help="minimum scaling factor for probabilities of initial states",
                               type=float,
                               default=0.2)
        rare.add_argument("--psi_max",
                               help="maximum scaling factor for probabilities of initial states",
                               type=float,
                               default=0.8)
        rare.add_argument(
            "--no_regret",
            help="deactivate regret approximation for ablation study",
            default=False,
            action="store_true"
        )

        debug = self.add_argument_group("debug")
        debug.add_argument(
            "--print_heatmaps",
            help="prints heatmaps during training which is useful for debugging",
            default=False,
            action="store_true",
        )
