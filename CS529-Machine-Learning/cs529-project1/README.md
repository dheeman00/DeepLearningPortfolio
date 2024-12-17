# Detection of Intron and Exon boundaries using ID3 Decision Trees
## Problem Description
### Splice Junctions
Splice junctions are points on a DNA sequence at which "superfluous" DNA is removed during the process of protein creation in higher organisms. The problem posed in this dataset is to recognize, given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing) and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks: recognizing exon/intron boundaries (referred to as EI sites), and recognizing intron/exon boundaries (IE sites). (In the biological community, IE borders are referred to as "acceptors" while EI borders are referred to as "donors".)

### Attributes predicted
Given a 60 base-pair DNA sequence, decide if the middle position of the sequence is:
  A. "intron -> exon" boundary (IE) [These are sometimes called "donors"]
  B. "exon -> intron" boundary (EI) [These are sometimes called "acceptors"]
  C. neither (N)

## Instructions
  1. Implement the ID3 decision tree learner, as described in Chapter 3 of Mitchell.  For initial debugging, it is recommended that you construct a very simple data set (e.g., based on a boolean formula) and test your program on it.
  2. Implement Information Gain with Gini-index, Entropy, and Miss-classification error, for evaluation criterion.
  3. Implement split stopping using the chi-square test. 
  4. Use your algorithm to train a decision tree classifier using the training data and report accuracy on the validation data. Compare accuracies by varying the evaluation criteria and confidence level in determining split stopping. For the latter, use 99%, 95% and 0% (i.e., you always grow the full tree).
  5. Submit the following through UNM Learn. 

Your code. Your code should contain appropriate comments to facilitate understanding. If appropriate, your code must contain a Makefile or an executable script that receives the paths to the training and testing files and a README file. For didactic purposes, you may upload snippets of your code in this competition after the deadline. 

A report (pdf format) that describes: A high-level description of how your code works. The accuracies you obtain under various settings. Explain which options work well and why. If all your accuracies are low, tell us what you have tried to improve them and what you suspect is failing. 

You can use any programming language of your choice You need to implement all the algorithms by yourself (do not use libraries or already established functions to calculate entropy, information gain, chi-square test, etc).

## Plan
### Language
Python

### Data
Convert the training data to numerical representation using Pandas and Numpy. Each position *i* in the DNA sequence is considered an attribute a<sub>*i*</sub> with values A,C,G,T,D,N,S corresponding to numerical values 0,1,2,3,4,5,6 respectively.

#### Data Structures
Our data is represented by 2D Numpy array where each row is a training example. The first column is the sample ID which shall be ignored in training and prediction. The last column in the training data is the label of the sample.

##### Node
The primary class used for the ID3 Decition Tree is Node. A node represents the parent class for the following subclasses:
+ Node: A DataNode represents the internal nodes of the tree. Each node must specify the decision Attribute and all children nodes associated with it. A DataNode cannot be a leaf node in the tree.
+ LeafNode: A ClassNode represnts a classification by the tree. Each node must specify a class label associated with the node. A ClassNode can only be a leaf in the tree.

### Functions and Classes
main.py
+ get\_label(data_set: numpy.ndarray) -> int
+ build\_tree(data\_set: numpy.ndarray, attributes: Set[int], error\_tolerance: float = 0.0, cost\_function: Callable[[np.ndarray], float] = None, chi2\_confidence: float = 0.95, value\_list: Union[Dict[str, int], List[str]] = None, class\_list: Union[Dict[str, int], List[str]] = None) -> Node
+ classify(node: Node, sample: numpy.ndarray) -> int
+ write\_output(node: Node, data_set: numpy.ndarray) -> None
+ max\_depth(node: Node) -> int

data.\_\_init\_\_.py
+ read\_file(path, samples\_dict: Dict[str, int] = None, labels\_dict: Dict[str, int] = None, ignore\_unknown: bool = False, labeled: bool = True, remove\_id: bool = True, conversion\_strat: int = 0) -> numpy.ndarray
+ \_convert\_samples(input\_list: List[str], dictionary: Dict[str, int] = None, ignore\_unknown: bool = False) -> numpy.ndarray
+ \_convert\_labels(input\_list: List[str], dictionary: Dict[str, int] = None, ignore\_unknown: bool = False) -> numpy.ndarray
+ convert\_to\_labels(data\_set: List[int], dictionary: Dict[int, str] = None, ignore\_unknown: bool = False) -> List[str]

stats.\_\_init\_\_.py
+ gain(data\_set: numpy.ndarray, attribute: int, cost\_function: Callable[[numpy.ndarray], float]) -> (float, Dict)
+ chi\_square(data\_set: numpy.ndarray, attribute: int, alpha: float) -> bool

stats.environment.py
+ set\_value\_list(values: List[int]) -> None
+ set\_class\_list(classes: List[int]) -> None

stats.purity.\_\_init\_\_.py
+ mce(data\_set: numpy.ndarray) -> float
+ entropy(data\_set: numpy.ndarray) -> float
+ gini(data\_set: numpy.ndarray) -> float

tree.\_\_init\_\_.py
+ class Node(object)
  + add\_child(key: int, value: Node) -> None
+ class LeafNode(Node)

#### Data Correction Function
The handling of the data for values which are not a single 'A', 'C', 'G', 'T' will be done using the following strategies:
+ Uniform: The attribute value is selected uniformly at random from the applicable 'A', 'C', 'G', 'T' value for the given symbol.
+ Max: The attribute value is the maximum applicable 'A', 'C', 'G', 'T' value of the feature in the entire training set.
+ Probabilistic: The attribute value is selected probabilitically using the distribution of applicable 'A', 'C', 'G', 'T' value of the feature in the entire training set. (Not implemented yet)
