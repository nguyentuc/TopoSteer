import torch
from typing import List
from dataset import UniDataset
import os
from utils import *
import json
from globalenv import *
import numpy as np 

def compute_perplexity_on_prompt(
    model,
    input_data,
    max_length: int = 512
):
    """
    Compute perplexity on the input prompt itself
    
    Args:
        model: Model wrapper
        input_data: Input tensor/data from dataset
        max_length: Maximum sequence length
        
    Returns:
        perplexity: Perplexity score
    """
    input_ids = input_data.to(model.device)
    with torch.no_grad():
        outputs = model.model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


def compute_perplexity_on_generation(
    model,
    input_data,
    generated_text: str,
    max_length: int = 200,
    batch_index: int = 0  # NEW: which sample in the batch
):
    """
    Compute perplexity on generated text GIVEN the prompt context.
    
    Args:
        model: Model wrapper
        input_data: Input tensor or BatchEncoding (the prompt, already tokenized)
        generated_text: Generated output text (the answer)
        max_length: Maximum sequence length to evaluate
        batch_index: Index of the sample in the batch (if batched)
        
    Returns:
        perplexity: Perplexity score for the generated text
    """
    # Handle empty generation
    if not generated_text or len(generated_text.strip()) == 0:
        return float('inf')
    
    # Tokenize ONLY the generated answer
    answer_tokens = model.tokenizer(
        generated_text, 
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False
    )
    answer_ids = answer_tokens['input_ids'].to(model.device)
    
    # Handle case where tokenization produces empty result
    if answer_ids.shape[1] == 0:
        return float('inf')
    
    # Extract prompt token IDs - handle both Tensor and BatchEncoding
    if isinstance(input_data, dict) or hasattr(input_data, 'input_ids'):
        prompt_ids = input_data['input_ids'].to(model.device)
    else:
        prompt_ids = input_data.to(model.device)
    
    # Handle batched input - extract the specific sample
    if prompt_ids.dim() == 2 and prompt_ids.shape[0] > 1:
        # Batched input - select the specific sample
        prompt_ids = prompt_ids[batch_index:batch_index+1]  # Keep 2D: [1, seq_len]
    elif prompt_ids.dim() == 1:
        # 1D tensor - add batch dimension
        prompt_ids = prompt_ids.unsqueeze(0)
    
    # Now both should be [1, seq_len]
    # Concatenate prompt + answer token IDs directly
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    
    # Truncate if too long
    if full_ids.shape[1] > max_length:
        full_ids = full_ids[:, :max_length]
    
    # Create labels: -100 for prompt tokens, actual IDs for answer tokens
    labels = full_ids.clone()
    prompt_length = prompt_ids.shape[1]
    
    # Ensure we don't mask beyond sequence length
    if prompt_length >= full_ids.shape[1]:
        return float('inf')
    
    labels[:, :prompt_length] = -100  # Mask prompt
    
    # Verify we have answer tokens to evaluate
    num_answer_tokens = (labels != -100).sum().item()
    if num_answer_tokens == 0:
        return float('inf')
    
    # Create attention mask
    attention_mask = torch.ones_like(full_ids)
    
    # Get model output with loss computed ONLY on answer tokens
    with torch.no_grad():
        try:
            outputs = model.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return float('inf')
    
    # Check for invalid loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        return float('inf')
    
    # Compute perplexity
    perplexity = torch.exp(loss).item()
    
    # Debug output for unreasonably high perplexity
    if perplexity > 1000:
        print(f"\n=== WARNING: Very High Perplexity ===")
        print(f"PPL: {perplexity:.2f} | Loss: {loss.item():.4f}")
        print(f"Prompt tokens: {prompt_length} | Answer tokens: {num_answer_tokens}")
        print(f"Generated text: '{generated_text[:100]}'")
        print(f"=====================================\n")
    return perplexity

def get_perplexity_BASE_results(
    model,
    test_dataset: UniDataset,
    max_new_tokens: int = 200,
):
    """..."""
    assert test_dataset.train == False, "Only Use Test Mode"
    assert test_dataset.set in ["test", "val"], "Only Use Test Dataset"
    
    model.reset_all()
    
    all_perplexities = []
    
    # Generate and compute perplexity for each sample
    for d in test_dataset:
        # Generate text
        cur_results = model.generate(d.to(model.device), max_new_tokens=max_new_tokens)
        
        # Compute perplexity on each generated text
        for idx, generated_text in enumerate(cur_results):
            try:
                ppl = compute_perplexity_on_generation(model, d, generated_text, batch_index=idx)
                all_perplexities.append(ppl)
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue
    
    avg_perplexity = np.mean(all_perplexities) if all_perplexities else float('inf')
    print(f"Baseline Perplexity: {avg_perplexity:.4f}")
    return avg_perplexity


def get_perplexity_results(
    model,
    layers: List[int],
    test_dataset: UniDataset,
    Alphas: List[float],
    train_task: str,
    train_method: str,
    max_new_tokens: int = 200,
):
    """
    Compute perplexity with steering applied
    
    Args:
        model: Model wrapper
        layers: List of layers to apply steering
        test_dataset: Test dataset
        Alphas: Steering strengths for each layer
        train_task: Task name for loading vectors
        train_method: Method for loading vectors
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        avg_perplexity: Average perplexity with steering
        perplexities: List of per-sample perplexities
    """
    assert test_dataset.train == False, "Only Use Test Dataset"
    assert test_dataset.set in ["val", "test"], "Only Use Val and Test Dataset"
    assert len(layers) == len(Alphas), "layers, Alphas must have same length"
    
    vec_root = f"./Vectors/{train_task}/{train_method}"
    
    vects = {}
    # Load steering vectors
    for i in range(len(layers)):
        vector_path = vec_root + f"/L{layers[i]}.pt"
        vects[layers[i]] = torch.load(vector_path).to(model.device)
        vects[layers[i]] *= Alphas[i]
    
    # Apply steering to model
    model.reset_all()
    for l in layers:
        model.set_add_activations(l, vects[l])
    
    all_perplexities = []
    
    # Generate and compute perplexity for each sample
    for d in test_dataset:
        # Generate text with steering
        cur_results = model.generate(d.to(model.device), max_new_tokens=max_new_tokens)
        
        # Compute perplexity on each generated text
        for idx, generated_text in enumerate(cur_results):
            try:
                ppl = compute_perplexity_on_generation(model, d, generated_text, batch_index=idx)
                all_perplexities.append(ppl)
            except:
                print(f"Skipping sample due to error: {e}")
                continue
    
    avg_perplexity = np.mean(all_perplexities) if all_perplexities else float('inf')
    print(f"Steered Perplexity: {avg_perplexity:.4f}")
    return avg_perplexity

# All functions are about *test* dataset
def get_raw_BASE_results(
    model,
    test_dataset:UniDataset,
    max_new_tokens:int=200,
):
    assert test_dataset.train == False, "Only Use Test Mode"
    assert test_dataset.set in ["test","val"], "Only Use Test Dataset"


    # Results Save Path
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}")


    save_root = f"./Results/{test_dataset.task}-{test_dataset.set}/"
    save_path = f"{save_root}base.json"

    # Get Text Generation
    model.reset_all()
    results = []
    for d in test_dataset:
        cur_results = model.generate(d.to(model.device),max_new_tokens=max_new_tokens) # raw results
        results.extend(cur_results)
    with open(save_path,"w") as f:
        json.dump(results,f,indent=4)

    # Get Logits
    last_logits = []
    for d in test_dataset:
        cur_last_logits = model.get_last_logits(d.to(model.device)).detach().cpu()
        last_logits.extend(cur_last_logits)
    last_logits = torch.stack(last_logits,dim=0)
    last_logits = torch.softmax(last_logits,dim=-1)
    last_logits = last_logits[:,[QUICK_TOKEN_ID_DICT[" Yes"],QUICK_TOKEN_ID_DICT[" No"]]]
    keys = test_dataset.keys
    total = 0
    assert len(keys) == len(last_logits)
    for i in range(len(keys)):
        if keys[i] == 1:
            total += last_logits[i,0].item()/(last_logits[i,0].item()+last_logits[i,1].item())
        else:
            total += last_logits[i,1].item()/(last_logits[i,0].item()+last_logits[i,1].item())
    print(f"Prob: {total/len(keys)}")

    return total/len(keys)

def get_raw_results(
    model,
    layers:List[int],
    test_dataset:UniDataset,
    Alphas:List[float],
    train_task : str,
    train_method:str,
    max_new_tokens:int=200,
):
    assert test_dataset.train == False, "Only Use Test Dataset"
    assert test_dataset.set in ["val","test"], "Only Use Val and Test Dataset"
    assert len(layers) == len(Alphas), "layers, Alphas must have same length"


    vec_root = f"./Vectors/{train_task}/{train_method}"

    svec_path = vec_root[10:] # remove "./Vectors/"
    svec_path = svec_path.replace("/","+")

    vects = {}
    # read steering vector from the set of layer that need to steer
    for i in range(len(layers)):
        vector_path = vec_root + f"/L{layers[i]}.pt"
        vects[layers[i]] = torch.load(vector_path).to(model.device)
        vects[layers[i]] *= Alphas[i]


    # Results Save Path
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}")
    

    save_root = f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}/"
    file_name = "Res"
    for i in range(len(layers)):
        file_name += f"_L{layers[i]}x{Alphas[i]*100:.0f}"
    save_path = f"{save_root}{file_name}.json"


    # Add the steering vector to the model
    model.reset_all()
    for l in layers:
        model.set_add_activations(l,vects[l])

    # After steering, inference on the test dataset and compute the avg log prob
    results = []
    for d in test_dataset:
        cur_results = model.generate(d.to(model.device),max_new_tokens=max_new_tokens) # raw results
        results.extend(cur_results)

    with open(save_path,"w") as f:
        json.dump(results,f,indent=4)
    
    # Get Logits
    last_logits = []
    for d in test_dataset:
        cur_last_logits = model.get_last_logits(d.to(model.device)).detach().cpu()
        last_logits.extend(cur_last_logits)
    last_logits = torch.stack(last_logits,dim=0)
    last_logits = torch.softmax(last_logits,dim=-1)
    last_logits = last_logits[:,[QUICK_TOKEN_ID_DICT[" Yes"],QUICK_TOKEN_ID_DICT[" No"]]]
    keys = test_dataset.keys #get the key and compute the prob of align with this key
    total = 0
    assert len(keys) == len(last_logits)
    for i in range(len(keys)):
        if keys[i] == 1:
            total += last_logits[i,0].item()/(last_logits[i,0].item()+last_logits[i,1].item())
        else:
            total += last_logits[i,1].item()/(last_logits[i,0].item()+last_logits[i,1].item())
    print(f"Prob: {total/len(keys)}")

    return total/len(keys)