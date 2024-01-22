from transformers.training_args import OptimizerNames

def adjust_training_args(training_args, additional_args=None):

    # train
    training_args.optim = OptimizerNames.ADAFACTOR

    # save ckpt
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = 'rougeLsum'
    training_args.greater_is_better = True
    training_args.evaluation_strategy = 'steps'
    training_args.eval_steps = training_args.save_steps

    return training_args
