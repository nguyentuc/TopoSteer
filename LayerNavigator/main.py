from model_wrapper import LlamaWrapper,QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from strategy import UniStrategy

if __name__ == "__main__":
    ### Step 1.
    ### Complement the model path to the first line of the *globalenv.py* file.

    ### Step 2.
    ### Run the following code to get the main results in our paper.


    # Get Model
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    # Run for all the task that will be used to evaluate in the paper
    for task in Anth_MAIN:
        print(f"############## Task: {task} ################")

        # Get Base Results
        test_dataset = UniDataset(
            task = task,
            train = False,
            set = "test",
        )
        base_prob = get_raw_BASE_results(
            model=model,
            test_dataset=test_dataset,
        )
        print(f"Base Prob: {base_prob}")

        # Get Vectors: Implements steering vector extraction using Mean Difference or PCA methods, and saves activations used during computation.
        train_dataset = UniDataset(
            task = task,
            train = True,
            set = "train",
        )
        
        # extract mean different steering vector: LAYERS in range(32)
        uni_generate_vectors(
            method="md",
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )

        # Get Score (Main Component): Computes the steerability score for each layer based on activations and steering vectors.
        get_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
        )

        # Get LayerNavigtor Results with different number of steering layers num_layers
        for num_layers in [1,3,5]:
            # rank the layer by s score for next step of choosing which layer to steer
            strategy = UniStrategy(
                task=task,
                strategy="my",
                num_layers=num_layers,
                method="md",
            )
            # apply steering vector to the model and compute the prob on the test set
            test_prob = get_raw_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0]*num_layers, # the strength of the steering vector
                train_task=task,
                train_method="md",
            )
            print(f"Layers: {num_layers}, Test Prob: {test_prob}")
        
        del test_dataset, train_dataset

