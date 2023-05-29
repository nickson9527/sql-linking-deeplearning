# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:46:43 2023

@author: 309
"""

import csv
import json
import mysql.connector
from sklearn.preprocessing import MinMaxScaler

with open('../hw4/hw4_config.json') as f:
    config = json.load(f)

host = config['host']
user = config['user']
passwd = config['passwd']

mydb = mysql.connector.connect(host=host, user=user, passwd=passwd)
mycursor = mydb.cursor()
mycursor.execute("DROP DATABASE IF EXISTS DB_traing")
mycursor.execute("CREATE DATABASE IF NOT EXISTS DB_traing")
mycursor.execute("USE DB_traing")

# 建立名為 training_data 的資料表
mycursor.execute("CREATE TABLE IF NOT EXISTS training_data (id INT AUTO_INCREMENT PRIMARY KEY, time VARCHAR(50), Shihimen INT, Feitsui INT, TPB INT, inflow INT, outflow INT, Feitsui_outflow INT, Tide INT, TPB_level INT)")

with open('./較少的數據集(測試用).csv', newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # 跳過標題行
    for row in reader:
        time = row[0] if row[0] else None
        Shihimen = round(float(row[1]), 2) if row[1] else None
        Feitsui = round(float(row[2]), 2) if row[2] else None
        TPB = round(float(row[3]), 2) if row[3] else None
        inflow = round(float(row[4]), 2) if row[4] else None
        outflow = round(float(row[5]), 2) if row[5] else None
        Feitsui_outflow = round(float(row[6]), 2) if row[6] else None
        Tide = round(float(row[7]), 2) if row[7] else None
        TPB_level = round(float(row[8]), 2) if row[8] else None
        sql = "INSERT INTO training_data (time, Shihimen, Feitsui, TPB, inflow, outflow, Feitsui_outflow, Tide, TPB_level) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (time, Shihimen, Feitsui, TPB, inflow, outflow, Feitsui_outflow, Tide, TPB_level)
        mycursor.execute(sql, val)
        

        
# 建立名為 typhoon_list 的資料表
mycursor.execute("CREATE TABLE IF NOT EXISTS typhoon_list (id INT AUTO_INCREMENT PRIMARY KEY, time VARCHAR(50), name VARCHAR(50), startTime VARCHAR(50), endTime VARCHAR(50))")

with open('./颱風場次.csv', newline='',encoding="utf-8") as csvfile: 
    reader = csv.reader(csvfile)
    next(reader) # 跳過標題行
    for row in reader:
        time = row[0] if row[0] else None
        name = row[1] if row[1] else None
        startTime = row[2] if row[2] else None
        endTime = row[3] if row[3] else None
        sql = "INSERT INTO typhoon_list (id, time, name, startTime, endTime) VALUES (NULL, %s, %s, %s, %s)"
        val = (time, name, startTime, endTime)
        mycursor.execute(sql, val)


#%%抓取資料

mydb.commit()

typhoon_name = '麥德姆'  ############################## 輸入要擷取數據的颱風名稱
selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide', 'TPB_level']

# 從 typhoon_list 資料表中查詢指定颱風名稱的起始時間和結束時間
mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE name = %s", (typhoon_name,))
result = mycursor.fetchone()
if result:
    start_time = result[0]
    end_time = result[1]


# column_mapping = {
#     'Shihimen': 'normalized_Shihimen',
#     'Feitsui': 'normalized_Feitsui',
#     'TPB': 'normalized_TPB',
#     'inflow': 'normalized_inflow',
#     'outflow': 'normalized_outflow',
#     'Feitsui_outflow': 'normalized_Feitsui_outflow',
#     'Tide': 'normalized_Tide',
#     'TPB_level': 'normalized_TPB_level'
# }

# # 建立欄位名稱的映射表，將原始欄位名稱映射為規範化後的欄位名稱
# column_mapping = {
#     'Shihimen': 'normalized_Shihimen',
#     'Feitsui': 'normalized_Feitsui',
#     'TPB': 'normalized_TPB',
#     'inflow': 'normalized_inflow',
#     'outflow': 'normalized_outflow',
#     'Feitsui_outflow': 'normalized_Feitsui_outflow',
#     'Tide': 'normalized_Tide',
#     'TPB_level': 'normalized_TPB_level'
# }


# columns = ', '.join([
#     f"({col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {column_mapping[col]}"
#     for col in selected_columns
# ])

# # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化
# query = f"""
#     SELECT time, {columns}
#     FROM training_data AS t
#     CROSS JOIN (
#         SELECT
#             {', '.join([f"MIN({col}) AS min_{col}, MAX({col}) AS max_{col}" for col in selected_columns])}
#         FROM training_data
#     ) AS minmax
#     WHERE time >= %s AND time <= %s
# """

# params = (start_time, end_time)
# mycursor.execute(query, params)
# result = mycursor.fetchall()


# for row in result:
#     time = row[0]
#     normalized_row = [time] + [float(val) if val is not None else None for val in row[1:]]
#     print([f"{val:.4f}" if isinstance(val, float) else val for val in normalized_row])

selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'TPB_level']
column_mapping = {
    'Shihimen':         'coeff_Shihimen',
    'Feitsui':          'coeff_Feitsui',
    'TPB':              'coeff_TPB',
    'inflow':           'coeff_inflow',
    'outflow':          'coeff_outflow',
    'Feitsui_outflow':  'coeff_Feitsui_outflow',
    'TPB_level':        'coeff_TPB_level'
}
label = 'Tide'

columns = ', '.join([
    f"(COUNT({col})*SUM({col}*{label}) - SUM({col})*SUM({label})) / SQRT((COUNT({col})*SUM(POWER({col}, 2))-POWER(SUM({col}),2))*(COUNT({label})*SUM(POWER({label}, 2))-POWER(SUM({label}),2)))"
    for col in selected_columns
])

# 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化
query = f"""
    SELECT {columns}
    FROM training_data
    WHERE time >= %s AND time <= %s
"""

params = (start_time, end_time)
mycursor.execute(query, params)
result = mycursor.fetchall()


for row in result:
    # time = row[0]
    # normalized_row = [time] + [float(val) if val is not None else None for val in row[1:]]
    # print([f"{val:.4f}" if isinstance(val, float) else val for val in normalized_row])
    print(row)
#%%CHECk用來檢查資料有沒有乖乖存進去
# select_query = "SELECT * FROM training_data"
# mycursor.execute(select_query)
# result = mycursor.fetchall()
# for row in result:
#     print(row)
# select_query = "SELECT * FROM typhoon_list"
# mycursor.execute(select_query)
# result = mycursor.fetchall()

# for row in result:
#     print(row)
# #%%
# with open('C:\hw4/hw4_config.json') as f:
#     config = json.load(f)

# host = config['host']
# user = config['user']
# passwd = config['passwd']

# mydb = mysql.connector.connect(host=host, user=user, passwd=passwd)
# mycursor = mydb.cursor()
# mycursor.execute("CREATE DATABASE IF NOT EXISTS DB_traing")
# mycursor.execute("USE DB_traing")

mycursor.execute("DROP DATABASE IF EXISTS DB_traing")

print("Database deleted successfully.")