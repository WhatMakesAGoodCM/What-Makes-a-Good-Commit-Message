import MySQLdb
from urllib.request import urlopen
from urllib.request import Request

import json


db = MySQLdb.Connect(host='localhost', port=3306, user='root', passwd='******', db='github_repos',
                     charset='utf8')
cursor = db.cursor()
# db table: changes of OOS project
fileTable = 'file_junit4'
# db table: commit messages of OOS project
messageTable = 'message_junit4'

def cmp(elem):
    return elem[0]


def filter_tokens(length, tokens, tags):
    indices = []
    tokens = tokens.split(' ')
    tags = tags.split(' ')
    for i in range(1, length):
        if str(tokens[i]).startswith('@'):
            indices.append(i)
        elif str(tokens[i]).isalnum() and not str(tokens[i]).islower():
            if str(tags[i]).startswith("NN"):
                # if str(tokens[i]) == 'file_name' or str(tokens[i]) == 'version':
                #     continue
                indices.append(i)
            else:
                before = i>0 and str(tokens[i-1])=="'"
                after = i+1<len(tokens) and str(tokens[i+1]) == "'"
                if before and after:
                    indices.append(i)

    return indices, tokens


def request(url):
    url = url.replace('https://github.com/', 'https://api.github.com/repos/').replace('/commit/', '/commits/')
    try:
        req = Request(url + '?access_token=adcd53eed06c848fd3cc5ceab67056545f214a53')
        response = urlopen(req).read()
        commit = json.loads(response.decode())
        files = commit['files']
        return files
    except Exception as e:
        print(e)
        return None


def search_in_patches(url, indices, tokens):
    patches = []
    files = request(url)
    while files is None:
        files = request(url)
    for file in files:
        if 'patch' in file.keys():
            patch = file['patch']
            patches.append(patch)
    fount_indices = []
    found_tokens = []
    for index in indices:
        for patch in patches:
            if str(patch).find(tokens[index]) > -1:
                if index>0 and index<len(tokens)-1 and str(tokens[index-1])=="'" and str(tokens[index+1])=="'":
                    found_tokens.append("'" + str(tokens[index]) + "'")
                else:
                    found_tokens.append(tokens[index])
                fount_indices.append(index)
                break

    return fount_indices, list(set(found_tokens))


def escape(message, replacement):
    start = 0
    escapes = []
    index = str(message).find(replacement, start, len(message))
    while index > -1:
        escapes.append([index, index + len(replacement)])
        start = index + len(replacement)
        index = str(message).find(replacement, start, len(message))
    return escapes


def get_unreplacable(message, replacement):
    unreplacable_indices = []
    start = 0
    index = str(message).find(replacement, start, len(message))
    while index > -1:
        start = index + len(replacement)
        for i in range(index, start):
            unreplacable_indices.append(i)
        index = str(message).find(replacement, start, len(message))
    return unreplacable_indices


def replace_tokens(message, tokens):
    unreplacable = []
    replacements = ['<file_name>', '<version>', '<url>', '<enter>', '<tab>','<issue_link>', '<pr_link>', '<otherCommit_link>','<method_name>']
    for replacement in replacements:
        unreplacable += get_unreplacable(message, replacement)

    # find out start and end index of replaced tokens
    locations = []
    for t in tokens:
        end = 0
        while end < len(message):
            start = str(message).find(t, end, len(message))
            if start == -1:
                break
            end = start + len(t)
            before = start > 0 and str(message[start - 1]).isalnum()
            after = end < len(message) and str(message[end]).isalnum()
            if not before and not after:
                locations.append([start, end])

    # 合并互相包含的被替换token的区间
    locations.sort(key=cmp)
    i = 0
    while i < len(locations) - 1:
        if locations[i][1] > locations[i + 1][0]:
            if locations[i][0] == locations[i + 1][0]:
                if locations[i][1] < locations[i + 1][1]:
                    locations.pop(i)
                elif locations[i][1] > locations[i + 1][1]:
                    locations.pop(i + 1)
            elif locations[i][0] < locations[i + 1][0] and locations[i][1] >= locations[i + 1][1]:
                locations.pop(i + 1)
        else:
            i += 1

    # merge continuous replaced tokens
    new_locations = []
    i = 0
    start = -1
    while i < len(locations):
        if start < 0:
            start = locations[i][0]
        if i < len(locations) - 1 and locations[i + 1][0] - locations[i][1] < 2:
            i += 1
            continue
        else:
            end = locations[i][1]
            new_locations.append([start, end])
            start = -1
            i += 1

    # replace tokens in message with <file_name>
    end = 0
    new_message = ""
    for location in new_locations:
        start = location[0]
        new_message += message[end:start]
        new_message += "<iden>"
        end = location[1]
    new_message += message[end:len(message)]

    return new_message


def update_new_message1(id, new_message):
    sql = "update "+ messageTable +" set new_message1 = '%s' where id = %d" % (new_message.replace("'", "''"), id)
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()


def get_dataset():
    sql = "select id, url, new_message, allennlp_len, allennlp_tokens, allennlp_tags from " + messageTable +\
          " where new_message is not null and allennlp_len is not null and id >=18500 and id <19500"
    cursor.execute(sql)
    rows = cursor.fetchall()
    return rows

# todo:Added #finished(long nanos, Description description),其中的方法名无法识别
if __name__ == "__main__":
    dataset = get_dataset()
    for row in dataset:
        id = row[0]
        url = row[1]
        new_message = row[2]
        length = row[3]
        tokens = row[4]
        tags = row[5]
        if len(new_message) > 0:
            indices, tokens = filter_tokens(length, tokens, tags)
            if len(indices) > 0:
                fount_indices, found_tokens = search_in_patches(url, indices, tokens)
                if len(fount_indices) > 0:
                    new_message = replace_tokens(new_message, found_tokens)

        new_message.replace('<enter>', '$enter').replace('<tab>', '$tab').\
        replace('<url>', '$url').replace('<version>', '$versionNumber')\
        .replace('<pr_link>','$pullRequestLink>').replace('<issue_link >','$issueLink')\
        .replace('<otherCommit_link>','$otherCommitLink').replace("<method_name>","$methodName")\
        .replace("<file_name>","$fileName").replace("<iden>","$token")

        update_new_message1(id, new_message)

    db.close()
