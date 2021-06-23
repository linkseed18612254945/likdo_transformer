import requests
import json

headers = {'content-type': "application/json"}


def post_dict_data(url, data_dict):
    """
    发送post请求，包含一个字典数据的json的请求体
    :param url: string， post请求地址
    :param data_dict: dict, 请求体数据
    :return: response, 请求返回结果
    """
    data_json = json.dumps(data_dict)
    response = requests.post(url, data=data_json, headers=headers)
    # response = json.loads(response)
    return response


def post_dict_return_dict_data(url, data_dict):
    """
    发送post请求，包含一个字典数据的json的请求体
    :param url: string， post请求地址
    :param data_dict: dict, 请求体数据
    :return: response, 请求返回结果
    """
    data_json = json.dumps(data_dict)
    response = requests.post(url, data=data_json, headers=headers)

    response = json.loads(response.text)
    # print(response)
    return response


# def post_dict_return_dict_data(url, data_dict):
#     """
#     发送post请求，包含一个字典数据的json的请求体
#     :param url: string， post请求地址
#     :param data_dict: dict, 请求体数据
#     :return: response, 请求返回结果
#     """
#     # data_json = json.dumps(data_dict)
#     response = requests.post(url, data_dict, headers=headers)
#     response = json.loads(response.text)
#     return response


def get_dict_data(url, data_dict):
    response = requests.get(url, params=data_dict)
    return response