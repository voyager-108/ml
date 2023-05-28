from typing import Union
import torch
import numpy as np
from _collections_abc import Sequence
from .OSG.osg import osg
import logging

def ascending_samples(l: Sequence, min_index: int=0, n: int=None) -> list[list]:
    """Given an iterable, this function returns all the possible ordered samples that satisfy the following conditions:
        1. of size 'n'
        2. The indices of the elements extracted are at least 'min_index'
        3. The index of the i-th element in the original list is less or equal to the index of the (i + 1) -th element in the original list.
            
        for a sequence [1, 2, 3], min_index = 0 and sample size 2, the possible combinations are: 
            [1, 1], [1, 2], [1, 3], [2, 2],  [2, 3], [3, 3]
        # we can see that the ordered sample [2, 1] is not valid as the index of value 2 is larger that index of the value 1 in the origina list

    Args:
        l (Sequence): The original seqeuence
        min_index (int, optional): The minimum index in the original sequence to start sampling from. Defaults to 0.
        n (int, optional): The size of the sample. Defaults to None and then to the length of the sequence

    Returns:
        list[list]: All possible ordered samples satisfying the conditions mentioned above.
    """
    
    if n is None:
        n = len(l)

    # base cases: n == 0 or min_index is the last index
    if n == 0:
        return [[]]

    if min_index == len(l) - 1:
        return [[l[-1] for _ in range(n)]]

    result = []
    for i in range(n + 1):
        temp = [l[min_index] for _ in range(i)]
        temp_res = ascending_samples(l, min_index= min_index + 1, n=n-len(temp))
        result.extend([temp + s for s in temp_res])

    return result



def __combination_score(group: Union[np.ndarray, torch.tensor], boundaries: list[int], classes: list) -> float:
    """Given: 
        1. a 2D matrix representing the probabilities assigned to N predefined elements
        2. boundaries specifying groups of elements with the same class
        3. The class assigned to each group
    This function evaluates estimates the quality of the combination of boundaries and classes according to the probabilities in the probabilites parameter
    
    Args:
        group (Union[np.ndarray, torch.tensor]): the probabilities matrix
        boundaries (list[int]): the list of indices representing the boundaries in the "group" parameter 
        classes (list): the class of each block / group

    Returns:
        float: The score of the combination
    """

    if not isinstance(group, np.ndarray):
        # the input is supposed to be either a tensor or a numpy array
        group = group.cpu().numpy()# # .squeeze() # convert the tensor to numpy array and remove any extra dimensions
    
    # make sure to proceed only with numpy arrays
    assert isinstance(group, (np.ndarray)), "THE INPUT MUST CONVERTED TO A NUMPY ARRAY"
    assert len(boundaries) == len(classes), "THE NUMBER OF CLASSES AND BOUNDARIES MUST BE EQUAL"

    
    assert len(boundaries) == len(classes)

    bs = boundaries.copy()  
    cs = classes.copy()

    # the initial "classes" paramter can have consecutive blocks with identical classes
    # this creates a larger block and thus the boundaries and the classes parameter should be reconstructed 

    if len(classes) >= 1:
        for i, c in enumerate(cs[1:]):
            if c == cs[i]:
                bs[i] = None
                cs[i] = None

    # now we have our lists in the correct format (after removing the redundant boundaries and classes marked as None)
    bs = [b for b in bs if b is not None]
    cs = [c for c in cs if c is not None]

    # make sure both lists are of the same size
    assert len(cs) == len(bs), "FILTERING THE BOUNDARIES OR THE CLASSES WENT WRONG !!"

    score = 0
    start_index = 0
    for b, c in zip(bs, cs):
        score += np.mean(group[start_index: b + 1, c])
        start_index = b + 1

    return score / len(cs)


def __best_group_combination(group: Union[torch.tensor, np.ndarray], horizontal: bool = True) -> list[int]:
    """Given a probabilities matrix, we extract the class for each element (argmax)
        and the boundaries of groups of elements with the same class.
        
    Args:
        group (Union[torch.tensor, np.ndarray]): the probabilities matrix

    Returns:
        list[int]: the modified classes for each element in the given group
    """
    if not isinstance(group, np.ndarray):
        # the input is supposed to be either a tensor or a numpy array
        group = group.cpu().numpy()# .squeeze() # convert the tensor to numpy array and remove any extra dimensions
    
    # make sure to proceed only with numpy arrays
    assert isinstance(group, (np.ndarray)), "THE INPUT MUST CONVERTED TO A NUMPY ARRAY"

    if not horizontal:
        # the code assumes the rows represent the elements
        group = group.T

    # extract the number of classes in the group
    last_element_class = np.argmax(group[0])
    boundaries, classes = [], []

    # determine the initial blocks and their associated classes
    for index, element in enumerate(group):
        element_class = np.argmax(element)
        # if the element's class is differnet than the last one update
        if element_class != last_element_class:
            # add the previous index as the last element of the previous segment
            boundaries.append(index - 1)
            # add the class of the previous segment
            classes.append(last_element_class)
            # ipdb.set_trace(context=10)
            # update the last_element_class
            last_element_class = element_class

    # always append the last_element_class and the last index
    classes.append(last_element_class)
    boundaries.append(len(group) - 1)

    # out of all possible samples, the function ascending samples generates all the possible orders of classes
    # that can possibly reach the optimal cost 

    all_perms = ascending_samples(classes)
    best_score, best_perm = None, None
    for p in all_perms:
        s = __combination_score(group, boundaries, p)
        if best_score is None or s > best_score:
            best_score = s
            best_perm = p

    # set the predictions according to the best permutation of classes
    start_index = 0
    block_preds = []

    for b, c in zip(boundaries, best_perm):
        block_preds.extend([c for _ in range(start_index, b + 1)])
        start_index = b + 1

    # this is the optimal prediction for each element in the initial probabilities matrix
    return block_preds    


def __combine(classifier_output: Union[torch.tensor, np.ndarray], boundaries: list[int], logits: bool = True) -> list[int]:
    """Given: 
        1. the output of a certain classifier: either the probabilities of the logits corresponding to each element in a sequence
        2. the output of the optimal sequential grouping in the format of boundaries
       build a more robust (less fragmented) predictions for the given sequence, by optimizing a cost function 
       based on the model's input for each block/group determined by the OSG algorithm.
    
    Args:
        classifier_output (Union[torch.tensor, np.ndarray]): a matrix that represents a model's output for a hidden sequence 
        boundaries (list[int]): the boundaries determined by the OSG algorithm for the same 
        logits (bool, optional): _description_. Defaults to True.
    
    Returns:
        list[int]: the final predictions
    """
    if logits:
        # first step is to convert the logits to probabilities using softmax
        softmax = torch.nn.Softmax(dim=-1)
        probabilities = softmax(classifier_output)
    else:
        probabilities = classifier_output
    
    start_index = 0
    predictions = []

    for b in boundaries:
        # extract the group data
        group = probabilities[start_index: b + 1]
        # update the start index
        start_index = b + 1
        predictions.extend(__best_group_combination(group))
    
    return predictions


def predict(embeddings: Union[np.ndarray, torch.Tensor], classifier_output: Union[torch.tensor, np.ndarray], logits: bool = True) -> list[int]:
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.cpu().numpy()# .squeeze()

    # determine the boundaries according to the OSG algorithm
    logger = logging.getLogger("ServerApplication").info("start calculating boundaries")
    boundaries = osg(embeddings)
    logger = logging.getLogger("ServerApplication").info("start combining")
    return  __combine(classifier_output=classifier_output, boundaries=boundaries, logits=logits)



