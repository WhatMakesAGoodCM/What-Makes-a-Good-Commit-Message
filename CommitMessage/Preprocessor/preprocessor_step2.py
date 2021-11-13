from nltk import word_tokenize
from nltk import pos_tag
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import MySQLdb


db = MySQLdb.Connect(host='localhost', port=3306, user='root', passwd='******', db='github_repos',
                     charset='utf8')
cursor = db.cursor()
# db table: changes of OOS project
fileTable = 'file_junit4'
# db table: commit messages of OOS project
messageTable = 'message_junit4'

def nltk_tag(message):
    words = word_tokenize(message)
    message = ' '.join(words)
    message = message.replace('< method_name >','<method_name>').replace('< file_name >', '<file_name>')\
        .replace('< url >', '<url>').replace('< version >', '<version>')
    words = message.split(' ')

    tags = pos_tag(words)
    tokens = [tag[0] for tag in tags]
    tags = [tag[1] for tag in tags]
    tokens = ' '.join(tokens)
    tags = ' '.join(tags)
    print(tokens)
    print(tags)
    return tags


def allennlp_tag(message, predictor):
    result = predictor.predict(message)
    tokens = result['tokens']
    tags = result['pos_tags']

    indices = []
    for i in range(len(tokens)):
        s = str(tokens[i])
        if s.startswith('file_name>') or s.startswith('version>') or s.startswith('url>') \
                or s.startswith('enter>') or s.startswith('tab>') or s.startswith('iden>') or s.startswith('method_name>')\
                or s.startswith('pr_link>') or s.startswith('issue_link>') or s.startswith('otherCommit_link>'):
            indices.append(i)
        elif s.endswith('<file_name') or s.endswith('<version') or s.endswith('<url') \
                or s.endswith('<enter') or s.endswith('<tab') or s.endswith('<iden') or s.endswith('<method_name')\
                or s.endswith('<pr_link') or s.endswith('<issue_link') or s.endswith('<otherCommit_link'):
            indices.append(i)

    new_tokens = []
    new_tags = []
    for i in range(len(tokens)):
        if i in indices:
            s = str(tokens[i])
            if s.startswith('file_name>'):
                s = s.replace('file_name>', '')
                new_tokens.append('file_name')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('method_name>'):
                s = s.replace('method_name>', '')
                new_tokens.append('method_name')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('version>'):
                s = s.replace('version>', '')
                new_tokens.append('version')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('url>'):
                s = s.replace('url>', '')
                new_tokens.append('url')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('enter>'):
                s = s.replace('enter>', '')
                new_tokens.append('enter')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('tab>'):
                s = s.replace('tab>', '')
                new_tokens.append('tab')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('iden>'):
                s = s.replace('iden>', '')
                new_tokens.append('iden')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('pr_link>'):
                s = s.replace('pr_link>', '')
                new_tokens.append('pr_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('issue_link>'):
                s = s.replace('issue_link>', '')
                new_tokens.append('issue_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.startswith('otherCommit_link>'):
                s = s.replace('otherCommit_link>', '')
                new_tokens.append('otherCommit_link')
                new_tags.append('XX')
                new_tokens.append('>')
                new_tags.append('XX')
                new_tokens.append(s)
                new_tags.append('XX')
            elif s.endswith('<file_name'):
                s = s.replace('<file_name', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('file_name')
                new_tags.append('XX')
            elif s.endswith('<method_name'):
                s = s.replace('<method_name', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('method_name')
                new_tags.append('XX')
            elif s.endswith('<version'):
                s = s.replace('<version', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('version')
                new_tags.append('XX')
            elif s.endswith('<url'):
                s = s.replace('<url', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('url')
                new_tags.append('XX')
            elif s.endswith('<enter'):
                s = s.replace('<enter', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('enter')
                new_tags.append('XX')
            elif s.endswith('<tab'):
                s = s.replace('<tab', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('tab')
                new_tags.append('XX')
            elif s.endswith('<iden'):
                s = s.replace('<iden', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('iden')
                new_tags.append('XX')
            elif s.endswith('<pr_link'):
                s = s.replace('<pr_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('pr_link')
                new_tags.append('XX')
            elif s.endswith('<issue_link'):
                s = s.replace('<issue_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('issue_link')
                new_tags.append('XX')
            elif s.endswith('<otherCommit_link'):
                s = s.replace('<otherCommit_link', '')
                new_tokens.append(s)
                new_tags.append('XX')
                new_tokens.append('<')
                new_tags.append('XX')
                new_tokens.append('otherCommit_link')
                new_tags.append('XX')
        else:
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])
    tokens = new_tokens
    tags = new_tags
    length = len(tokens)

    new_tokens = []
    new_tags = []
    targets = ['file_name', 'version', 'url', 'enter', 'tab', 'iden', 'issue_link', 'pr_link', 'otherCommit_link','method_name']
    i = 0
    while i < length:
        if i < length-2 and tokens[i] == '<' and tokens[i+1] in targets and tokens[i+2] == '>':
            new_tokens.append(tokens[i] + tokens[i+1] + tokens[i+2])
            new_tags.append('XX')
            i += 3
        else:
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])
            i += 1

    tokens = new_tokens
    tags = new_tags
    length = len(tokens)
    # 使用空格连接tokens
    tokens = ' '.join(tokens)
    tags = ' '.join(tags)
    print('----------------------------------------------------------------------')
    print(tokens)
    print(tags)
    # print(trees)
    return tokens, tags, length


def update_tags(cursor, db, tokens, tags, length, id):
    sql = "update "+ messageTable +" set allennlp_len = %d, allennlp_tokens = '%s', allennlp_tags = '%s' where id = %d" % \
          (length, tokens.replace("'", "''"), tags.replace("'", "''"), id)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print(sql)
        db.rollback()


def get_data():
    sql = "select id, new_message from "+ messageTable +" where new_message is not null " \
          + "and allennlp_len is null"
    # 重新更新差异数据
    # sql = "select a.id, a.new_message from " + messageTable + " a, trainset.message_train train where a.new_message is not null " \
    #       + " and a.url = train.url and a.new_message != train.new_message"

    cursor.execute(sql)
    rows = cursor.fetchall()
    return rows


if __name__ == "__main__":
    # -1代表cpu运行，需要用到句子中的短语结构就用constituent parse成分句法分析 ，而需要用到词与词之间的依赖关系就用dependency parse依存句法分析。
    archive = load_archive('./tools/elmo-constituency-parser-2018.03.14.tar.gz', -1)
    predictor = Predictor.from_archive(archive, 'constituency-parser')

    data = get_data()
    for row in data:
        message = row[1]
        tokens, tags, length = allennlp_tag(message, predictor)
        update_tags(cursor, db, tokens, tags, length, row[0])

    db.close()
