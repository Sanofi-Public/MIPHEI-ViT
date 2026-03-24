from .evaluators.dataset_evaluator import DATASET_EVALUATOR_BASES
from .evaluators.model_evaluator import MODEL_EVALUATOR_BASES


def get_evaluator_class(model_name: str, dataset_name: str):
    model_base = MODEL_EVALUATOR_BASES[model_name]
    dataset_base = DATASET_EVALUATOR_BASES[dataset_name]
    class_name = f"{dataset_name.capitalize()}{model_name.capitalize()}Evaluator"
    return type(class_name, (model_base, dataset_base), {})


def evaluate(args):
    EvaluatorClass = get_evaluator_class(args.model, args.dataset)
    print(f"Evaluator: {EvaluatorClass.__name__}")

    kwargs = {
        "checkpoint_dir": args.checkpoint_dir,
        "config_dir": args.config_dir,
        "min_area": args.min_area,
        "save_logreg": args.save_logreg
    }
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.num_workers is not None:
        kwargs["num_workers"] = args.num_workers
    if args.model == "rosie":
        if args.pred_dir is None:
            raise ValueError("Need to specify a pred directory")
        kwargs.update({"pred_dir": args.pred_dir})

    evaluator = EvaluatorClass(
        **kwargs
    )
    evaluator.evaluate()
