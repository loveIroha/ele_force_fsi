import requests
from datetime import date
from cryptography.fernet import Fernet
import json
import sys

# 打印脚本名称
print("路径为:", sys.argv[1])
test_path = sys.argv[1]
import os
import glob

def find_json_files(test_path):
    # 构造文件搜索模式
    search_pattern = os.path.join(test_path, "*output.json")

    # 使用 glob 模块查找符合模式的文件
    matching_files = glob.glob(search_pattern)

    # 输出找到的文件列表
    print("Found files:")
    for file in matching_files:
        print(file)

    return matching_files


# 1. 加密解密，用于存储token
def encrypt_message(message, key):
    cipher_suite = Fernet(key)
    encrypted_message = cipher_suite.encrypt(message.encode())
    return encrypted_message

def decrypt_message(encrypted_message, key):
    cipher_suite = Fernet(key)
    decrypted_message = cipher_suite.decrypt(encrypted_message).decode()
    return decrypted_message

key = b'5sXanMEWkWvSxmXy4BSiQt_uosLHdojpZWHGSKwmM9E='  
encrypted_token = b'gAAAAABlwyiwtbc74DVuSIoMLyu6J6ZctKrbp3o1JiV70DmqTD8XXJFQh9HZzeXqM6OTwnIa8wC5Bu-rcI3edICw-xSAZPFAm7WKUjcWZJrw8vfpWJmwLwBizE2xoudLoRZsP6UsOVvpNrEQ5RraZ1bdb5TbX51RtA=='
token = decrypt_message(encrypted_token, key)

# 2. 查询数据库，获取数据格式
database_id = '20b7f09150bb48bf8e7a7b2852e2fa83'
headers = {
    'Notion-Version': '2021-05-13',# 在新版中必须加入版本信息
    'Authorization': 'Bearer '+token,# 这一行也必须要有
}
url_notion = 'https://api.notion.com/v1/databases/'+database_id+'/query'
notion_response = requests.post(url_notion,headers=headers)
notion_data     = notion_response.json()['results']

json_files = find_json_files(test_path)
print(json_files)

for json_file in json_files:
    # 3. 读取运行日志
    with open(json_file, 'r') as file:
        # 从文件中加载 JSON 数据
        json_data = json.load(file)

    # 4. 构建上传数据
    body = {
        'parent': {'type': 'database_id', 'database_id': database_id},
    }
    body['properties'] = notion_data[0]['properties']
    body['properties']['stamp']['rich_text']=[{ "type": "text", "text": { "content": json_data['stamp'] } }]
    body['properties']['git commit']['rich_text']=[{ "type": "text", "text": { "content": json_data['GIT_COMMIT_HASH'] } }]
    body['properties']['git date']['rich_text']=[{ "type": "text", "text": { "content": json_data['GIT_COMMIT_DATE'] } }]
    body['properties']['git version']['rich_text']=[{ "type": "text", "text": { "content": json_data['GIT_VERSION'] } }]

    def update_properties(body_properties, tag, avg, cnt, max, min):
        body_properties['标题']['title'][0]['text']['content']=tag
        body_properties['测试日期']['date']['start']=date.today().strftime("%Y-%m-%d")
        body_properties['avg']['number']=avg
        body_properties['cnt']['number']=cnt
        body_properties['max']['number']=max
        body_properties['min']['number']=min

    # 5. 上传数据
    url_notion_additem = 'https://api.notion.com/v1/pages'
    for i in range(len(json_data['data'])):
        tag = json_data['data'][i]
        update_properties(body['properties'], tag, json_data[tag]['avg'],json_data[tag]['cnt'],json_data[tag]['max'],json_data[tag]['min'])
        notion_additem = requests.post(url_notion_additem,headers=headers,json=body)
        if notion_additem.status_code == 200:
            print('成功')
        elif notion_additem.status_code == 429:
            print('频率限制')
        elif notion_additem.status_code == 400:
            print('失败')

print('https://faceted-fireplace-be0.notion.site/20b7f09150bb48bf8e7a7b2852e2fa83?v=344ad6b9279f480b8bdb26af2867f7eb')
# 使用限制：https://developers.notion.com/reference/request-limits
# 官方文档：https://developers.notion.com/docs/working-with-page-content#reading-nested-blocks