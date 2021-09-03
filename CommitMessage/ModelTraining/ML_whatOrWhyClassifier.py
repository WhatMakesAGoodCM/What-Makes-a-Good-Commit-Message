import os
import warnings

import numpy as np
import pandas as pd
import torch
import transformers as ppb

from ClassifierTraining import trainClassifierNegative
from logger import mylog

warnings.filterwarnings('ignore')


def BertEmbedding(labeledDF):
    # 加载bert模型
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    # Tokenization
    tokenized = labeledDF['new_message1'].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=150)))
    # padding
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # masking
    attention_mask = np.where(padded != 0, 1, 0)

    logger.info("===== getting features ======")
    # embedding
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    # 将bert输出的第一层作为句子的特征向量
    return last_hidden_states[0][:, 0, :].numpy()


if __name__ == "__main__":
    global logger
    logger = mylog()

    df = pd.read_csv("~/message_sample.csv")

    labeledDF = df[df.label.notnull() & df.if_mulit_commit.isnull()]
    verifiedDf = df[df.label.isnull()]

    labeledDF["new_message1"].apply(lambda x: x.replace('<enter>', '$enter').replace('<tab>', '$tab'). \
                                    replace('<url>', '$url').replace('<version>', '$versionNumber') \
                                    .replace('<pr_link>', '$pullRequestLink>').replace('<issue_link >', '$issueLink') \
                                    .replace('<otherCommit_link>', '$otherCommitLink').replace("<method_name>",
                                                                                               "$methodName") \
                                    .replace("<file_name>", "$fileName").replace("<iden>", "$token"))

    # 包含what和why的是负样本，标记为0
    whyLabels = labeledDF['label'].apply(
        lambda x: 0 if x == 0 else (1 if x == 1.0 else (0 if x == 2.0 else (1 if x == 3.0 else 0))))
    whatLabels = labeledDF['label'].apply(
        lambda x: 0 if x == 0 else (1 if x == 1.0 else (1 if x == 2.0 else (0 if x == 3.0 else 1))))
    Labels = labeledDF['label'].apply(
        lambda x: 1 if x == 0 else (0 if x == 1.0 else (0 if x == 2.0 else (0 if x == 3.0 else 1))))

    idList = labeledDF['id']
    repoIdList = labeledDF['repo_id']
    try:
        features = np.load("./feature.npy")
    except IOError:
        print("feature file don't exit.")
        features = BertEmbedding(labeledDF)
        np.save("feature.npy", features)
        print("message features have been saved. shape: " + str(features.shape))

    logger.info("ADASYN_ReSampling+Ten_Fold")
    print("==========================whyInfor===============================")
    logger.info("=========================whyInfor=============================")
    trainClassifierNegative(features, whyLabels, idList, repoIdList,
                            str(os.path.split(__file__)[-1].split(".")[0][7:]) + "_why", logger)
    print("=========================whatInfor===============================")
    logger.info("=========================whatInfor======================")
    trainClassifierNegative(features, whatLabels, idList, repoIdList,
                            str(os.path.split(__file__)[-1].split(".")[0][7:]) + "_what", logger)
