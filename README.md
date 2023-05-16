# Multi-label Text Classification using Fine-tuning of BERT uncased
<br>
This repository provides an example implementation of multi-label text classification using fine-tuning of the BERT uncased model. It demonstrates how to train a BERT-based model on a custom dataset with multiple labels and make predictions on new text inputs.
<br>

## Requirements
To run the code in this repository, the following dependencies are required:

* Python 3.x
* PyTorch
* Transformers library (Hugging Face) 4.28
* pandas
* numpy
* scikit-learn
<br>

# Dataset Preparation

1. Extracting specified attributes from json files 
2. cleaning the to_fill and stories dataframes
3. Tokenization: Using a tokenizer from the Transformers library to tokenize the text data
4. Creating a custom dataset

# Model Training 

1. Initialize the BERT uncased model: Load the BERT uncased model from the Transformers library, ensuring that it is compatible with multi-label classification
2. Initialize the training arguments
3. Instantiate the Trainer
4. Train the model
5. Save the trained model

# Additional Resources 
For further information and resources on multi-label text classification and BERT fine-tuning, consider the following:

* Hugging Face Transformers library documentation: https://huggingface.co/transformers/ . 
" BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - Original BERT paper by Devlin et al.
" A Gentle Introduction to BERT" - Article explaining BERT in detail by Jacob Devlin: https://towardsdatascience.com/a-gentle-introduction-to-bert-96b596fdbc1d
* Scikit-learn documentation for multi-label classification: https://scikit-learn.org/stable/modules/multiclass.html
