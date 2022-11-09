import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true") 
parser.add_argument("--num_prototypes", type=int, default=40)
parser.add_argument("--num_pos_prototypes", type=int, default=40)
parser.add_argument("--modelname", type = str)
parser.add_argument("--data_dir", type = str)
parser.add_argument("--model", type=str, default="ProtoTEx")
parser.add_argument("--model_checkpoint", type=str, default=None)

args = parser.parse_args()

datasets_config =  {
    "data/logical_fallacy_with_none": {
        'features': {
            'text': 'source_article', 
            'label': 'updated_label'
        },
        'classes': {
            'O': 0,
            'ad hominem': 1,
            'ad populum': 2,
            'appeal to emotion': 3,
            'circular reasoning': 4,
            'fallacy of credibility': 5,
            'fallacy of extension': 6,
            'fallacy of logic': 7,
            'fallacy of relevance': 8,
            'false causality': 9,
            'false dilemma': 10,
            'faulty generalization': 11,
            'intentional': 12,
            'equivocation': 13
        }
    },
    "data/logical_fallacy_augmented": {
        'features': {
            'text': 'text', 
            'label': 'label'
        },
        'classes': {
            'ad hominem': 0,
            'ad populum': 1,
            'appeal to emotion': 2,
            'circular reasoning': 3,
            'fallacy of credibility': 4,
            'fallacy of extension': 5,
            'fallacy of logic': 6,
            'fallacy of relevance': 7,
            'false causality': 8,
            'false dilemma': 9,
            'faulty generalization': 10,
            'intentional': 11,
            'equivocation': 12
        }
    },
    "data/finegrained": {
        'features': {
            'text': 'text', 
            'label': 'label'
        }, 
        'classes': {
            "fallacy of red herring": 0,
            "faulty generalization": 1, 
            "ad hominem": 2, 
            "false causality": 3,
            "circular reasoning": 4, 
            "ad populum": 5, 
            "fallacy of credibility": 6, 
            "appeal to emotion": 7, 
            "fallacy of logic": 8, 
            "intentional": 9, 
            "false dilemma": 10,
            "fallacy of extension": 11, 
            "equivocation": 12,
        }
    }, 
    "data/coarsegrained": {
        'features': {
            'text': 'text',
            'label': 'label_updated'
        },
        'classes': {
            "fallacy of relevance": 0,
            "fallacies of defective induction": 1,
            "fallacies of presumption": 2,
            "fallacy of ambiguity": 3,
        }
    },
    "data/bigbench": {
        'features': {
            'text': 'text',
            'label': 'label'
        }, 
        'classes': {
            0: 0,
            1: 1
        }
    },
    "data/ptc_slc_without_none_with_context/fine": {
        'features': {
            'text': 'text',
            'label': 'label'
        }, 
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
        }
    }
}

bad_classes = [
    "prejudicial language",
    "fallacy of slippery slope",
    "slothful induction"
]