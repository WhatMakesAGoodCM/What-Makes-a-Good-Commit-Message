import re
import MySQLdb



db = MySQLdb.Connect(host='localhost', port=3306, user='root', passwd='****', db='github_repos',
                     charset='utf8')
cursor = db.cursor()
# db table: changes of OOS project
fileTable = 'file_junit4'
# db table: commit messages of OOS project
messageTable = 'message_junit4'

def select_message():
    # 使用替换掉pr和issues链接的message进行下一步预处理
    sql = "select id, repo_id, label, n_label, url, messageIden, first_file_id, file_num from "+ messageTable \
            + " where messageIden is not null"
    cursor.execute(sql)
    rows = cursor.fetchall()
    return rows


def select_file(message_id):
    sql = "select file_name from "+ fileTable +" where message_id = " + str(message_id)
    cursor.execute(sql)
    rows = cursor.fetchall()
    return rows


def update_new_message(id, new_message):
    # 防止new_message过长而无法存储,去掉多余的空格
    sql = "update "+ messageTable +" set new_message = '%s' where id = %d" % (new_message.replace("'", "''").replace("   "," "), id)
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()


def update_new_message2(id,message,newMessage):
    sql = "insert compare_New_Message (id, url, message, old_new_message, new_message) values (%d,'%s','%s','%s','%s') "\
          %(id, '', message, '', new_message.replace("'", "''").replace("   "," "))
    print(sql)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()


def split(path):  # splitting by seperators, i.e. non-alnum
    # 输入：changes文件名（路径）或者message
    # 操作：按照非字母数字进行分割，并且忽略预处理得到的 ‘<xxx>’
    new_sentence = ''
    for s in path:
        if not str(s).isalnum():
            if len(new_sentence) > 0 and not new_sentence.endswith(' '):
                new_sentence += ' '
            if s != ' ':
                new_sentence += s
                new_sentence += ' '
        else:
            new_sentence += s
    tokens = new_sentence.replace('< enter >', '<enter>').replace('< tab >', '<tab>').\
        replace('< url >', '<url>').replace('< version >', '<version>')\
        .replace('< pr _ link >','<pr_link>').replace('< issue _ link >','<issue_link>')\
        .replace('< otherCommit_link >','<otherCommit_link>').strip().split(' ')
    return tokens


def find_url(message):
    if 'git-svn-id: ' in message:
        # 对于git-svn-id链接，单独处理
        pattern = re.compile(
            r'git-svn-id:\s+(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\s+(?:[a-z]|[0-9])+(?:-(?:[a-z]|[0-9])+){4})')
    else:
        pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(pattern, message)
    urls = sorted(list(set(urls)), reverse=True)
    for url in urls:
        message = message.replace(url, '<url>')
    return message


def find_version(message):
    pattern = re.compile(r'[vVr]?\d+(?:\.\w+)+(?:-(?:\w)*){1,2}')
    versions = pattern.findall(message)
    versions = sorted(list(set(versions)),reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')

    pattern2 = re.compile(r'[vVr]?\d+(?:\.\w+)+')
    versions = pattern2.findall(message)
    # 去除重复pattern
    versions = sorted(list(set(versions)),reverse=True)
    for version in versions:
        message = message.replace(version, '<version>')
    return message


def find_rawCode(message):
    rawCodeSta = message.find('```')
    replaceIden = []
    res = ''
    while rawCodeSta>0:
        rawCodeEnd = message.find('```', rawCodeSta + 3, len(message))
        if rawCodeEnd!=-1:
            replaceIden.append([rawCodeSta,rawCodeEnd+3])
        else:
            break
        rawCodeSta = message.find('```', rawCodeEnd + 3, len(message))
    if len(replaceIden)>0:
        end = 0
        for iden in replaceIden:
            res += message[end:iden[0]]
            end = iden[1]
        res += message[end:len(message)]
        return res
    else:
        return message


def find_SignInfo(message):
    index = message.find("Signed-off-by")
    if index==-1:
        return message
    if index>0 and (message[index-1]=='"' or message[index-1]=="'"):
        return message
    subMessage = message[index:]
    enterIndex = subMessage.find(">")
    message = message[0:index]+" "+message[index+enterIndex+1:]
    return message

def get_all_data():
    samples = []
    rows = select_message()
    for row in rows:
        id = row[0]
        message = row[5]
        file_names = select_file(id)
        # 将changes fileName进行分割
        # file_names = [split(file_name[0]) for file_name in file_names]
        file_names = [file_name[0] for file_name in file_names]

        message = find_url(message)
        message = find_version(message)
        message = find_rawCode(message)
        message = find_SignInfo(message)
        if message[0]==" ":
            isUnuse = True
            for s in message:
                if s.isalnum():
                    isUnuse=False
                    break
            if isUnuse:
                message = "empty log message"

        samples.append([id, message, file_names])
    print(len(samples))
    return samples


def tokenize(identifier):  # camel case splitting
    new_identifier = ""
    identifier = list(identifier)
    new_identifier += identifier[0]
    for i in range(1, len(identifier)):
        if str(identifier[i]).isupper() and (str(identifier[i-1]).islower() or (i < len(identifier)-1 and str(identifier[i+1]).islower())):
            if not new_identifier.endswith(" "):
                new_identifier += " "
        new_identifier += identifier[i]
        if str(identifier[i]).isdigit() and i < len(identifier)-1 and not str(identifier[i+1]).isdigit():
            if not new_identifier.endswith(" "):
                new_identifier += " "
    return new_identifier.split(" ")


def find_file_name2(sample):
    # 以下不处理非驼峰形式的changes file
    filePath = sample[2]
    messageOld = sample[1]
    message = messageOld.lower()
    replaceTokens = []
    otherMeanWords = ['version','test','assert','junit']
    specialWords = ['changelog','contributing','release','releasenote','readme','releasenotes']
    punctuations = [',', '.', '?', '!', ';', ':', '、']

    for file in filePath:
        #   以'/'分割
        filePathTokens = file.split('/')
        fileName = filePathTokens[-1]
        # 如果文件名以".md"结尾则不进行替换
        if fileName.endswith(".md"):
            continue
        # 直接包含文件名
        if fileName.lower() in message:
            index = message.find(fileName.lower())
            replaceTokens.append(messageOld[index:index+len(fileName)])
        if '.' in fileName:
            # 获取无后缀的文件名
            newFileName = fileName
            pattern = re.compile(r'(?:\d+(?:\.\w+)+)')
            versions = pattern.findall(newFileName)
            for version in versions:
                if version!=newFileName:
                    newFileName = newFileName.replace(version, '')
            # 去除后拓展名并全部小写，对于以'.'开头或者包含'.'的文件名，
            # 仅去除拓展名 e.g. ".Trivas.yml"->".Trivas"
            lastIndex = newFileName[1:].rfind('.')
            if lastIndex == -1:
                lastIndex = len(newFileName)-1
            newFileName = newFileName[:lastIndex+1]
            fileNameGreen = newFileName.lower()
            # 直接包含去掉后缀的文件名
            if fileNameGreen in specialWords:
                continue
            elif fileNameGreen in otherMeanWords:
                index = 0
                while index!=-1:
                    tempIndex = message[index+1:len(message)].find(fileNameGreen)
                    if tempIndex ==-1:
                        break
                    else:
                        index =index + 1 + tempIndex
                        if index!=-1 and messageOld[index].isupper():
                            replaceTokens.append(messageOld[index:index+len(fileNameGreen)])
                            break
            # msg包含不带拓展名的文件名，e.g. AClass.java in 'xxx AClss/method() xxx'
            elif fileNameGreen in message:
                index = message.find(fileNameGreen)
                replaceTokens.append(messageOld[index:index + len(fileNameGreen)])
            else:
                # 驼峰文件名，对应于msg中分开的连续单词
                fileNameTokens = tokenize(newFileName)
                if len(fileNameTokens) < 2:
                    continue
                if fileNameTokens[0].lower() in message:
                    camelSta = message.find(fileNameTokens[0].lower())
                    camelEnd = -1
                    tempMessag = message[camelSta:]
                    while camelSta >= 0 and len(tempMessag) > 0:
                        tempMessagTokens = tempMessag.split(' ')
                        find = True
                        if tempMessagTokens[0] == fileNameTokens[0].lower():
                            # 删除句号和逗号等标点符号，其他符号不可能对应于驼峰文件名
                            for i in range(0, min(len(tempMessagTokens),len(fileNameTokens))):
                                if len(tempMessagTokens[i])<2:
                                    continue
                                if str(tempMessagTokens[i][-1]) in punctuations:
                                    tempMessagTokens[i] = tempMessagTokens[i][:-1]

                            for i in range(0, len(fileNameTokens)):
                                if i < len(tempMessagTokens) and tempMessagTokens[i] != fileNameTokens[i].lower():
                                    find = False
                                    break
                                elif i > len(tempMessagTokens):
                                    find = False
                                    break
                            if find:
                                lastTokenIndex = tempMessag.find(fileNameTokens[-1].lower())
                                camelEnd = len(tempMessag[:lastTokenIndex]) + len(fileNameTokens[-1])+ camelSta
                                if camelEnd < len(tempMessag) and tempMessag[camelEnd] in punctuations:
                                    camelEnd += 1
                                break
                        index = message[camelSta + 1:].find(fileNameTokens[0].lower())
                        if index == -1:
                            break
                        camelSta = camelSta + 1 + index
                        tempMessag = message[camelSta:]
                    if camelSta!=-1 and camelEnd !=-1:
                        replaceTokens.append(messageOld[camelSta:camelEnd])
    replaceTokens = list(set(replaceTokens))
    return replaceTokens


def cmp(elem):
    return elem[0]


def replace_file_name(sample):
    # replaced_tokens = find_file_name(sample)
    replaced_tokens = find_file_name2(sample)
    message = sample[1]

    # find out start and end index of replaced tokens
    locations = []
    # 以'@' 开头的token 一般是annotation，并且通常会出现在patchs里，所以即使和文件名相同也要忽略
    diffMeanPunctuations = ['@']
    for t in replaced_tokens:
        end = 0
        while end<len(message):
            start = str(message).find(t, end, len(message))
            if start == -1:
                break
            end = start + len(t)
            before = start > 0 and (str(message[start-1]).isalnum() or str(message[start-1]) in diffMeanPunctuations)
            after = end < len(message) and str(message[end]).isalnum()
            if not before and not after:
               locations.append([start, end])

    # 合并互相包含的被替换token的区间
    locations.sort(key=cmp)
    i=0
    while i < len(locations)-1:
        if locations[i][1]>locations[i+1][0]:
            if locations[i][0]==locations[i+1][0]:
                if locations[i][1]<locations[i+1][1]:
                    locations.pop(i)
                elif locations[i][1]>locations[i+1][1]:
                    locations.pop(i+1)
            elif locations[i][0]<locations[i+1][0] and locations[i][1]>=locations[i+1][1]:
                locations.pop(i+1)
        else:
            i+=1

    # '.'和'#' 用于表示class中包含某个方法/字段，或者用于包路径,
    # eg. AClass.getInt()、FrameworkMethod#producesType()、org.junit.runner.Description#getTestClass
    backSymbols = ['.', '/']        #文件名之前的特殊符号
    forwardSymbols = ['.', '#']     #文件名之后的特殊符号
    newLocations = []
    newMethodeName = []

    for location in locations:
        sta = location[0]
        end = location[1]
        ifMethod = False
        packagePath = ''
        if sta>0 and str(message[sta-1]) in backSymbols:
            newSta = sta-1
            while newSta>=0 and str(message[newSta])!=' ':
                packagePath = str(message[newSta])+packagePath
                newSta-=1
            sta = newSta+1

        if end<len(message) and str(message[end]) in forwardSymbols:
            newEnd = end+1
            while newEnd<len(message) and str(message[newEnd])!=' ':
                newEnd+=1
            end = newEnd
            ifMethod = True
        if ifMethod:
            newMethodeName.append([sta, end])
        newLocations.append([sta, end])

        if packagePath != '':
            index = 0
            while index>=0:
                index = message.find(packagePath,index,len(message))
                if index == sta:
                    index = end
                elif index != -1:
                    indexEnd = index+len(packagePath)
                    while indexEnd< len(message) and str(message[indexEnd]) != " ":
                        indexEnd+=1
                    newLocations.append([index,indexEnd])
                    index+=1


    newLocations.sort(key=cmp)
    newMethodeName.sort(key=cmp)
    # replace tokens in message with <file_name>
    end = 0
    new_message = ""
    for location in newLocations:
        start = location[0]
        new_message += message[end:start]
        if location in newMethodeName:
            new_message += " <method_name> "
        else:
            new_message += " <file_name> "
        end = location[1]
    new_message += message[end:len(message)]

    return new_message


if __name__ == '__main__':
    samples = get_all_data()
    for sample in samples:
        if len(sample[1]) > 0:
            new_message = replace_file_name(sample)
            update_new_message(sample[0], new_message)
        else:
            update_new_message(sample[0], sample[1])
    db.close()
