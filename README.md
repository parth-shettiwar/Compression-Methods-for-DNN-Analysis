# An Analysis of Compression Methods for Deep learning networks
This project was done as part of course 259-Learning Machines under professor Tony Nowatzki. 
Team Members:  
1) Parth Shettiwar 
2) Madhav Shankar. 
## Major Contributions:  
* We present a small survey like analysis, in form of takeaway messages, of existing in literature Compression methods for deep learning networks. 

* Based on evaluation metrics, we perform an uniform basis comparison of these methods (1 from each section, discussed before) quantitatively to understand their applicability, use cases and various pros and cons depending on situation. 

* A design of a newer compression techniques, through ensembling and combination of the algorithms, to achieve better performance as compared to individual algorithms on CIFAR10 dataset. 

## Code Structure and commands
I. FOLDER: QUANTIZATION. 
* Static Quantization:  
Run the python notebook: StaticQuantization.ipynb. 

* Quantization Aware Training:  
Run the python notebook: QuantizationAwareTraining.ipynb. 

II. FOLDER: CUDA. 
* Hardware Implication of Quantization:  
FP32 Matrix Multiplication: nvcc classifier_Matrix.cu -o fp32. 
INT8 Matrix multiplication: nvcc int.cu -o int8. 

* Hardware Implication of Pruning (Sparse Matrices):  
Dense Matrix Vector Multiplication (Baseline): nvcc classifierMV.cu -o classifier.  
Sparse Matrix Vector Multiplication (Baseline): nvcc sparse.cu -o sparse. 
Dense Matrix Matrix Multiplication (Optimized): nvcc classifier_Matrix.cu -o fp32. 
Sparse Matrix Matrix Multiplication (CuSparse): nvcc cuSparse.cu -o cusparse -lcusparse.   
Then run nvprof on the generated object: nvprof ./sparse.   

III. FOLDER: knowledge-distillation-torch. 
* Knowledge distillation.   
Download CIFAR10 Dataset and add to folder knowledge-distillation-pytorch/data-cifar10.   
Load     
python train.py --model_dir experiments/cnn_distill.   
Here resnet18 is the default teacher network and cnn_distill is name of student model.   
To make any other model as student model, simply create a folder inside experiments with its name and add params.json in format present in cnn_distill. Dataset is cifar10. 
At same time inference is also carried out.   

* Knowledge Distillation Model generation:  
Copy KDModels.ipynb notebook to KD folder and run.  

* Ensemble of KD models:  
Copy Ensemble.ipynb notebook to KD folder and run.  

IV. FOLDER: lookahead-pruning. 
* Pruning:    
Download CIFAR10 Dataset and add to folder lookahead_pruning/dataset. 
python main.py --dataset cifar10 --network resnet18 --method mp. 
Here again base network is resnet18 and dataset is cifar10. 
The training will be done for all compression rates.  
At same time inference is also carried out. 

* Model size analysis of pruned models:  
python size.py. 
As mentioned in report, we had to modify the way weights are saved for a pruned model to know the exact model size. 
Just put the directory name, which has he pruned models and it will give prune model size, and number of parameters.   

V. Combination of models:  
All the combination of models was done in a sequential fashion. Follow the above steps for each of the compression technique sequential to get the desired combined compression model.  
