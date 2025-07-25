import time
import os
from basereal import BaseReal
from logger import logger

def llm_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    from openai import OpenAI
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 填写DashScope SDK的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': message}],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )
    result=""
    first = True
    for chunk in completion:
        if len(chunk.choices)>0:
            #print(chunk.choices[0].delta.content)
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")
                first = False
            msg = chunk.choices[0].delta.content
            lastpos=0
            #msglist = re.split('[,.!;:，。！?]',msg)
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；" :
                    result = result+msg[lastpos:i+1]
                    lastpos = i+1
                    if len(result)>10:
                        logger.info(result)
                        nerfreal.put_msg_txt(result)
                        result=""
            result = result+msg[lastpos:]
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    nerfreal.put_msg_txt(result)

def llm_wenda_response(message,nerfreal:BaseReal):
    #调用闻达大模型
    #http://127.0.0.1:17860/api/chat method POST
    #body内容 "{\"prompt\":\"你是智能百科,每个问题尽量不超过20字,回答内容不要带格式,问题如下:"+prompt+"\",\"keyword\":\"你是智能百科,每个问题尽量不超过20字,回答内容不要带格式,问题如下:"+prompt+"\",\"temperature\":0.8,\"top_p\":0.8,\"max_length\":4096,\"history\":[]}"
    start = time.perf_counter()
    import requests
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "prompt": "你是智能百科,每个问题尽量不超过20字,回答内容不要带格式,问题如下:"+message,
        "keyword": "你是智能百科,每个问题尽量不超过20字,回答内容不要带格式,问题如下:"+message,
        "temperature": 0.8,
        "top_p": 0.8,
        "max_length": 4096,
        "history": []
    }
    url = "http://127.0.0.1:17860/api/chat"
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(f'llm_wenda_response: {response.text}')
        result = response.text
        end = time.perf_counter()
        logger.info(f"llm_wenda Time to response: {end-start}s")
        nerfreal.put_msg_txt(result)
    else:
        logger.error(f"Error in llm_wenda_response: {response.status_code} - {response.text}")

def llm_java_wenda_response(message,nerfreal:BaseReal):
    start = time.perf_counter()
    import requests
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "prompt": message,
    }
    url = "http://127.0.0.1:9885/chatgpt/api/getWendaContent/zhonghang"
    response = requests.post(url, headers=headers, json=data)
    # logger.info(f'llm_java_wenda_response: {response.text}')
    if response.status_code == 200:
        # logger.info(f'llm_java_wenda_response: {response.text}')
        # print(f'llm_wenda_response: {response.text}')
        result = response.text
        end = time.perf_counter()
        logger.info(f"llm_wenda Time to response: {end - start}s")
        # 获取result中的resultStr
        try:
            result = response.json().get("resultStr", "")
        except ValueError:
            logger.error("Response is not a valid JSON")
            result = response.text
        nerfreal.set_result_msg(result)
        nerfreal.put_msg_txt(result)
    else:
        logger.error(f"Error in llm_wenda_response: {response.status_code} - {response.text}")