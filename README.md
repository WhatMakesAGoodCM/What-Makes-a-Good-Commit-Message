# What-Makes-a-Good-Commit-Message
This repository contains the main data and scripts used in 'What Makes a Good Commit Message'
## Dataset
The folder dataset contains the following files.
* literature survery.xlsx
    * It contains the data of 46 relevant literatures reviewed in this study (Section 3.2).

* questionnaire.csv
    * It is the questionnaire which sent to experienced contributors.
    * It contains three questions.
    * It also contains an example of the actual content of the email.
  
* posts list.xlsx
    * It contains all posts we studied in Sec. 3.2.

* sampled messages.csv
    * It contains meta-information of 1649 labeled commit messages.
    * label = 0 means a commit message contains "Why and What".
    * label = 1 means a commit message  contains "Neither Why nor What".
    * label = 2 means a commit message  contains "No What".
    * label = 3 means a commit message  contains "No Why".
    * if_mulit_commit = 1 means a commit is non-atomic.
  
* maintenance type and expression way.xlsx
    * It contains the results of our RQ2: the expression ways of Why and What, as well as links to maintenance types.
  
## CommitMessage (Scripts)
The folder scripts contains the following files.

* Preprocessor
  * It contains the preprocessing of commit message, including the replacement of token in message, etc.

* ModelTraining
  * It contains the code for our model training, that is, the implementation of different classification techniques.


