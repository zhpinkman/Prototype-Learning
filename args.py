import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true")
# parser.add_argument("--nli_dataset", help="check if the dataset is in nli
# format that has sentence1, sentence2, label", action="store_true")
parser.add_argument("--num_prototypes", type=int, default=50)
parser.add_argument("--model", type=str, default="ProtoTEx")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--modelname", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--model_checkpoint", type=str, default=None)
parser.add_argument("--use_max_length", action="store_true")

# Wandb parameters
parser.add_argument("--project", type=str)
parser.add_argument("--experiment", type=str)
parser.add_argument("--nli_intialization", type=str, default="Yes")
parser.add_argument("--none_class", type=str, default="No")
parser.add_argument("--curriculum", type=str, default="No")
parser.add_argument("--augmentation", type=str, default="No")
parser.add_argument("--architecture", type=str, default="BART")


args = parser.parse_args()

datasets_config = {
    "data/finegrained": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "fallacy of logic": 0,
            "circular reasoning": 1,
            "appeal to emotion": 2,
            "intentional": 3,
            "faulty generalization": 4,
            "fallacy of extension": 5,
            "false dilemma": 6,
            "ad populum": 7,
            "ad hominem": 8,
            "false causality": 9,
            "equivocation": 10,
            "fallacy of relevance": 11,
            "fallacy of credibility": 12,
        },
        "max_length": 128,
    },
    "data/logical_fallacy_with_none": {
        "type": "classification",
        "features": {"text": "source_article", "label": "updated_label"},
        "classes": {
            "O": 0,
            "ad hominem": 1,
            "ad populum": 2,
            "appeal to emotion": 3,
            "circular reasoning": 4,
            "fallacy of credibility": 5,
            "fallacy of extension": 6,
            "fallacy of logic": 7,
            "fallacy of relevance": 8,
            "false causality": 9,
            "false dilemma": 10,
            "faulty generalization": 11,
            "intentional": 12,
            "equivocation": 13,
        },
    },
    "data/logical_fallacy_augmented_with_none": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "O": 0,
            "ad hominem": 1,
            "ad populum": 2,
            "appeal to emotion": 3,
            "circular reasoning": 4,
            "fallacy of credibility": 5,
            "fallacy of extension": 6,
            "fallacy of logic": 7,
            "fallacy of relevance": 8,
            "false causality": 9,
            "false dilemma": 10,
            "faulty generalization": 11,
            "intentional": 12,
            "equivocation": 13,
        },
    },
    "data/finegrained_with_none": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "O": 0,
            "ad hominem": 1,
            "ad populum": 2,
            "appeal to emotion": 3,
            "circular reasoning": 4,
            "fallacy of credibility": 5,
            "fallacy of extension": 6,
            "fallacy of logic": 7,
            "fallacy of relevance": 8,
            "false causality": 9,
            "false dilemma": 10,
            "faulty generalization": 11,
            "intentional": 12,
            "equivocation": 13,
        },
    },
    "data/logical_climate_finegrained": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "O": 0,
            "ad hominem": 1,
            "ad populum": 2,
            "appeal to emotion": 3,
            "circular reasoning": 4,
            "fallacy of credibility": 5,
            "fallacy of extension": 6,
            "fallacy of logic": 7,
            "fallacy of relevance": 8,
            "false causality": 9,
            "false dilemma": 10,
            "faulty generalization": 11,
            "intentional": 12,
            "equivocation": 13,
        },
    },
    "data/coarsegrained_with_none": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "O": 0,
            "fallacy of relevance": 1,
            "fallacies of defective induction": 2,
            "fallacies of presumption": 3,
            "fallacy of ambiguity": 4,
        },
    },
    "data/logical_climate_coarsegrained": {
        "type": "classification",
        "features": {"text": "text", "label": "coarse_label"},
        "classes": {
            "O": 0,
            "fallacy of relevance": 1,
            "fallacies of defective induction": 2,
            "fallacies of presumption": 3,
            "fallacy of ambiguity": 4,
        },
    },
    "data/bigbench": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {0: 0, 1: 1},
    },
    "data/ptc_slc_without_none_with_context/fine": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "Appeal_to_Authority": 0,
            "Appeal_to_fear-prejudice": 1,
            "Bandwagon": 2,
            "Black-and-White_Fallacy": 3,
            "Causal_Oversimplification": 4,
            "Doubt": 5,
            "Exaggeration,Minimisation": 6,
            "Flag-Waving": 7,
            "Loaded_Language": 8,
            "Name_Calling,Labeling": 9,
            "Obfuscation,Intentional_Vagueness,Confusion": 10,
            "Red_Herring": 11,
            "Reductio_ad_hitlerum": 12,
            "Repetition": 13,
            "Slogans": 14,
            "Straw_Men": 15,
            "Thought-terminating_Cliches": 16,
            "Whataboutism": 17,
        },
    },
    "data/ptc_slc_with_context": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "O": 0,
            "Appeal_to_Authority": 1,
            "Appeal_to_fear-prejudice": 2,
            "Bandwagon": 3,
            "Black-and-White_Fallacy": 4,
            "Causal_Oversimplification": 5,
            "Doubt": 6,
            "Exaggeration,Minimisation": 7,
            "Flag-Waving": 8,
            "Loaded_Language": 9,
            "Name_Calling,Labeling": 10,
            "Obfuscation,Intentional_Vagueness,Confusion": 11,
            "Red_Herring": 12,
            "Reductio_ad_hitlerum": 13,
            "Repetition": 14,
            "Slogans": 15,
            "Straw_Men": 16,
            "Thought-terminating_Cliches": 17,
            "Whataboutism": 18,
        },
    },
    "data/ptc_slc_aug_without_none_with_context": {
        "type": "classification",
        "features": {"text": "text", "label": "label"},
        "classes": {
            "Appeal_to_Authority": 0,
            "Appeal_to_fear-prejudice": 1,
            "Bandwagon": 2,
            "Black-and-White_Fallacy": 3,
            "Causal_Oversimplification": 4,
            "Doubt": 5,
            "Exaggeration,Minimisation": 6,
            "Flag-Waving": 7,
            "Loaded_Language": 8,
            "Name_Calling,Labeling": 9,
            "Obfuscation,Intentional_Vagueness,Confusion": 10,
            "Red_Herring": 11,
            "Reductio_ad_hitlerum": 12,
            "Repetition": 13,
            "Slogans": 14,
            "Straw_Men": 15,
            "Thought-terminating_Cliches": 16,
            "Whataboutism": 17,
        },
    },
}

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction",
]
