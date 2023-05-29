# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:06:18 2023

@author: 309
"""


import csv
import json
import mysql.connector
from sklearn.preprocessing import MinMaxScaler


mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="309309"
)


#%%Part 1

#CREATE DATABASE
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS DB_training")
mycursor.execute("USE DB_training")

# 建立名為 training_data 的資料表
mycursor.execute("CREATE TABLE IF NOT EXISTS training_data (id INT AUTO_INCREMENT PRIMARY KEY, time VARCHAR(50), Shihimen INT, Feitsui INT, TPB INT, inflow INT, outflow INT, Feitsui_outflow INT, Tide INT, TPB_level INT)")

path = 'F:/02_碩班作業/碩一下/SQL/project/teamproject/'

with open( path + '完整數據集.csv', newline='', encoding="utf-8") as csvfile:
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

with open(path + '颱風場次.csv', newline='',encoding="utf-8") as csvfile: 
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
        
mydb.commit()

#%%抓取資料

def train_test_data(typhoon_ids):
    typhoon_data = []
    
    for typhoon_id in typhoon_ids:
        mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
        result = mycursor.fetchone()
        if result:
            start_time = result[0]
            end_time = result[1]
    
            # 建立欄位名稱的映射表，將原始欄位名稱映射為規範化後的欄位名稱
            column_mapping = {
                'Shihimen': 'normalized_Shihimen',
                'Feitsui': 'normalized_Feitsui',
                'TPB': 'normalized_TPB',
                'inflow': 'normalized_inflow',
                'outflow': 'normalized_outflow',
                'Feitsui_outflow': 'normalized_Feitsui_outflow',
                'Tide': 'normalized_Tide',
                'TPB_level': 'normalized_TPB_level'
            }
    
            columns = ', '.join([
                f"({col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {column_mapping[col]}"
                for col in selected_columns
            ])
    
            # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行正規化
            query = f"""
                SELECT time, {columns}
                FROM training_data AS t
                CROSS JOIN (
                    SELECT
                        {', '.join([f"MIN({col}) AS min_{col}, MAX({col}) AS max_{col}" for col in selected_columns])}
                    FROM training_data
                ) AS minmax
                WHERE time >= %s AND time <= %s
            """
    
            params = (start_time, end_time)
            mycursor.execute(query, params)
            result = mycursor.fetchall()
            
            sequences = []
            
            for entry in result:
                values = []
                for value in entry[1:]:
                    if value is not None:
                        values.append(float(value))
                    else:
                        values.append(float('nan'))
                sequences.append(values)
            
            transposed_sequences = list(map(list, zip(*sequences)))
            typhoon_data.append(transposed_sequences)
    return typhoon_data

#選擇訓練測試
train_typhoon_ids = ['1','23','37']  # 輸入要擷取數據的颱風名稱列表
test_typhoon_ids = ['37']

selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide', 'TPB_level']

train_data = train_test_data (train_typhoon_ids) 
test_data = train_test_data (test_typhoon_ids) 

#調整輸入因子時間步長
TimeList = [3,3,3,6,10]    
EndTimeList = [0,0,0,0,0]
def SplitMuti(data, timeList, endTimeList, TPlus):
    """切分多個不同時間步長資料"""
    import numpy as np
    usedata = data[0]
    maxtime = max(timeList) #最大步長時間
    x = []   #預測點的前 N 天的資料
    y = []   #預測點
    for t in range(maxtime, len(usedata[0])-TPlus-max(endTimeList)):  
        arg_tuple = ()
        for i in range(len(timeList)):
            arg_tuple += tuple(usedata[i][t-timeList[i]:t+1+endTimeList[i]]) #取到 t+N
        temp = np.hstack(arg_tuple) # T-Tstep ~ T....
        x.append(temp)
        y.append(usedata[-1][t+TPlus]) # T+N
    return x, y 
SplitMuti(test_data, TimeList, EndTimeList, 1)



mycursor.close()
mydb.close()

    
    

#%%

import csv
import json
import mysql.connector

with open('C:\hw4/hw4_config.json') as f:
    config = json.load(f)

host = config['host']
user = config['user']
passwd = config['passwd']

mydb = mysql.connector.connect(host=host, user=user, passwd=passwd)
mycursor = mydb.cursor()

# 删除已存在的数据库
mycursor.execute("DROP DATABASE IF EXISTS DB_training")

print("Database deleted successfully.")