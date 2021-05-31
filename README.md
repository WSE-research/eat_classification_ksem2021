# Improving Answer Type Classification Quality Through Combined Question Answering Datasets

## Abstract

Understanding what a person is asking via a question is one of the first steps that humans use to find the answer. 
The same is true for Question Answering (QA) systems. 
Hence, the quality of the expected answer type classifier (EAT) has a direct influence on QA quality. 
Many research papers are aiming at improving short text classification quality, however, there is a lack of focus on the impact of training data characteristics on the classification quality as well as effective reuse of datasets through their augmentation and combination.
In this work, we propose an approach of analyzing and improving the EAT classification quality via a combination of existing QA datasets. 
We provide 4 new question classification datasets based on several well-known QA datasets as well as the approach to unify its class taxonomy.
We made a sufficient amount of experiments to demonstrate several valuable insights related to the impact of training data characteristics on the classification quality.
Additionally, an embedding-based approach for automatic data labeling error detection is demonstrated.

## Quick Links

* Full Paper (TBD)
* [Derived Datasets for Expected Answer Type classification](https://github.com/Perevalov/eat_classification_ksem2021/tree/main/data/UnifiedSubclassDBpedia)
* [Experimental Results](https://github.com/Perevalov/eat_classification_ksem2021/tree/main/data/experimental_results)
* [Labeling Error Analysis Results](https://github.com/Perevalov/eat_classification_ksem2021/tree/main/data/error_analysis)

## Experimental settings

The hyperparameters were fixed as follows: `BATCH_SIZE := 16`, `MAX_LEN := 128`, `EPOCHS := 8` (with early stopping on not minimizing loss with `PATIENCE := 1`). 
Each set of the experiments was repeated 5 times and standard error of the averaged results was calculated. 
The experiments were executed on a server with the following characteristics: `GPU: 2x Tesla V100 SXM2 32 GB`, `CPU: 96x Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz`, and `RAM: 1.6 TB`. 
Average training time in seconds is 2045, 2130, and 1090 -- for each model respectively. 
Average inference time in seconds is 17, 19, and 10 -- for each model respectively.

## Cite

TBD
