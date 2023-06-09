from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import csv
import json
import mysql.connector
import torch
# from ga import GA
class TyphoonDataset(Dataset):
    def __init__(self,
                 config_path = '../hw4/hw4_config.json',
                 data_path = './較少的數據集(測試用).csv',
                 typhoon_path = './颱風場次.csv',
                 typhoon_ids = None,
                 selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],
                 label='TPB_level',
                 concat_n=None,
                 split = None,
                ):
        super(TyphoonDataset).__init__()
        self.config_path = config_path
        self.data_path = data_path
        self.typhoon_path = typhoon_path
        self.typhoon_ids = typhoon_ids
        self.selected_columns = selected_columns
        self.label = label
        self.concat_n = concat_n
        self.split = split
        # pass
        self.x = []
        self.y = []
        if self.config_path:
            self.create_db(self.config_path,self.split)
        
  
    def __len__(self):
        return len(self.x)
        # pass
  
    def __getitem__(self,idx):
        # fname = self.data[idx]
            
        return self.x[idx],self.y[idx]
        pass
    
    def create_db(self,config_path = '../hw4/hw4_config.json',split='train'):
        with open(config_path) as f:
            config = json.load(f)
        host = config['host']
        user = config['user']
        passwd = config['passwd']

        self.mydb = mysql.connector.connect(host=host, user=user, passwd=passwd)
        self.mycursor = self.mydb.cursor()
        self.mycursor.execute(f"USE DB_sql")
        
        # if self.data_path and self.typhoon_path:
        #     self.load_db(self.data_path,self.typhoon_path)
    
    def load_db(self,data_path = './較少的數據集(測試用).csv',typhoon_path='./颱風場次.csv'):
        self.mycursor.execute(f"DROP DATABASE IF EXISTS DB_sql")
        self.mycursor.execute(f"CREATE DATABASE IF NOT EXISTS DB_sql")
        self.mycursor.execute(f"USE DB_sql")
        self.mycursor.execute("CREATE TABLE IF NOT EXISTS training_data (id INT AUTO_INCREMENT PRIMARY KEY, time TIMESTAMP, Shihimen FLOAT, Feitsui FLOAT, TPB FLOAT, inflow FLOAT, outflow FLOAT, Feitsui_outflow FLOAT, Tide FLOAT, TPB_level FLOAT)")

        with open(data_path, newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # 跳過標題行
            for row in reader:
                time =              row[0] if row[0] else None
                Shihimen =          float(row[1])if row[1] else None
                Feitsui =           float(row[2])if row[2] else None
                TPB =               float(row[3])if row[3] else None
                inflow =            float(row[4])if row[4] else None
                outflow =           float(row[5])if row[5] else None
                Feitsui_outflow =   float(row[6])if row[6] else None
                Tide =              float(row[7])if row[7] else None
                TPB_level =         float(row[8])if row[8] else None
                
                sql = "INSERT INTO training_data (time, Shihimen, Feitsui, TPB, inflow, outflow, Feitsui_outflow, Tide, TPB_level) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (time, Shihimen, Feitsui, TPB, inflow, outflow, Feitsui_outflow, Tide, TPB_level)
                self.mycursor.execute(sql, val)
        self.mydb.commit()

                
        # 建立名為 typhoon_list 的資料表
        self.mycursor.execute("CREATE TABLE IF NOT EXISTS typhoon_list (id INT AUTO_INCREMENT PRIMARY KEY, time VARCHAR(50), name VARCHAR(50), startTime VARCHAR(50), endTime VARCHAR(50))")

        with open(typhoon_path, newline='',encoding="utf-8") as csvfile: 
            reader = csv.reader(csvfile)
            next(reader) # 跳過標題行
            for row in reader:
                time = row[0] if row[0] else None
                name = row[1] if row[1] else None
                startTime = row[2] if row[2] else None
                endTime = row[3] if row[3] else None
                sql = "INSERT INTO typhoon_list (id, time, name, startTime, endTime) VALUES (NULL, %s, %s, %s, %s)"
                val = (time, name, startTime, endTime)
                self.mycursor.execute(sql, val)

        self.mydb.commit()
        # if self.typhoon_ids and self.selected_columns and self.label:
        #     self.fetch_data(self.typhoon_ids, self.selected_columns, self.label)

    def fetch_data(self,typhoon_ids,selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],label='TPB_level',table="training_data"):
        typhoon_data = []
        # print('fetch',typhoon_ids,selected_columns)
        for typhoon_id in typhoon_ids:
            self.mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
            result = self.mycursor.fetchone()
            if result:
                start_time = result[0]
                end_time = result[1]
        
            columns = ', '.join([
                # f"({col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {f'normalized_{col}'}"
                f"{col}"
                for col in selected_columns
            ])
    
            # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行正規化
            if len(selected_columns) == 0:
                query = f"""
                    SELECT time, YEAR(time), MONTH(time), DAY(time), HOUR(time), {label}
                    FROM {table} AS t
                    WHERE time >= %s AND time <= %s
                """
            else:
                query = f"""
                    SELECT time, YEAR(time), MONTH(time), DAY(time), HOUR(time), {columns}, {label}
                    FROM {table} AS t
                    WHERE time >= %s AND time <= %s
                """
            params = (start_time, end_time)
            self.mycursor.execute(query, params)
            result = self.mycursor.fetchall()
            if len(result)==0:
                # print('empty')
                continue
            # print(result)
            # exit()
            sequences = []
            labels = []
            for entry in result:
                # print(entry)
                values = []
                if None in entry:
                    continue
                for value in entry[1:-1]:
                    if value is not None:
                        values.append(float(value))
                    else:
                        values.append(float('nan'))
                        # continue
                labels.append(float(entry[-1]))
                sequences.append(values)
            
            # transposed_sequences = list(map(list, zip(*sequences)))
            if len(sequences)==0 or len(labels)==0:
                continue
            typhoon_data.append([sequences,labels])
        self.raw_data = typhoon_data
        # print(len(typhoon_data[0]))
        # return typhoon_data
        if self.concat_n:
            self.concat_data(self.concat_n)
    def min_max_data(self,typhoon_ids,selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],label='TPB_level',split=None):
        if split == None:
            split = self.split
        # print(split)
        # print('min_max',typhoon_ids)
        columns = ', '.join([
                f"{col}"
                for col in selected_columns
            ])
        params = []
        condition = ""
        
        for typhoon_id in typhoon_ids:
            self.mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
            result = self.mycursor.fetchone()
            if result:
                start_time = result[0]
                end_time = result[1]
                cond = " (time >= %s AND time <= %s)"
                params.append(start_time)
                params.append(end_time)
                if condition == "":
                    condition += cond
                else:
                    condition += " OR"
                    condition += cond

        # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化  
        if len(selected_columns) != 0:
            self.mycursor.execute(f'DROP VIEW IF EXISTS specified_typhoon_{split}')
            query = f"""
                CREATE VIEW specified_typhoon_{split} AS (
                SELECT time, {columns}, {label}
                FROM training_data
                WHERE {condition} )
            """
        else:
            return
        self.mycursor.execute(query, params)
        self.mydb.commit()
        columns = ', '.join([
                f"({col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {f'{col}'}"
                # f"{col}"
                for col in selected_columns
            ])
        self.mycursor.execute(f'DROP VIEW IF EXISTS normalized_specified_typhoon_{split}')
        query = f"""
            CREATE VIEW normalized_specified_typhoon_{split} AS (
            SELECT time, {columns}, {label}
            FROM specified_typhoon_{split} as t
            CROSS JOIN (
                SELECT
                    {', '.join([f"MIN({col}) AS min_{col}, MAX({col}) AS max_{col}" for col in selected_columns])}
                FROM specified_typhoon_{split}
            ) AS minmax )
        """


        self.mycursor.execute(query)
        self.mydb.commit()

    def normalize_data(self,typhoon_ids,selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],label='TPB_level'):
        columns = ', '.join([
                # f"(t.{col} - MIN(t.{col})) / (MAX(t.{col}) - MIN(t.{col}))"
                f"{col}"
                # f"(t.{col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {f'normalized_{col}'}"
                for col in selected_columns
            ])
        params = []
        condition = ""
        
        for typhoon_id in typhoon_ids:
            self.mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
            result = self.mycursor.fetchone()
            if result:
                start_time = result[0]
                end_time = result[1]
                cond = " (time >= %s AND time <= %s)"
                params.append(start_time)
                params.append(end_time)
                if condition == "":
                    condition += cond
                else:
                    condition += " OR"
                    condition += cond

        # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化  
        # query = f"""
        #     CREATE VIEW specified_typhoon_data AS (
        #     SELECT {columns}
        #     FROM training_data
        #     WHERE {condition} )
        # """
        if len(selected_columns) != 0:
            query = f"""
                CREATE VIEW specified_typhoon_data AS (
                SELECT time, {columns}, {label}
                FROM training_data
                WHERE {condition} )
            """
        else:
            return
        self.mycursor.execute(query, params)
        self.mydb.commit()
        columns = ', '.join([
                # f"({col} - minmax.min_{col}) / (minmax.max_{col} - minmax.min_{col}) AS {f'{col}'}"
                # f"{col}"
                # f"(COUNT({col})*SUM({col}*{label}) - SUM({col})*SUM({label})) / SQRT((COUNT({col})*SUM(POWER({col}, 2))-POWER(SUM({col}),2))*(COUNT({label})*SUM(POWER({label}, 2))-POWER(SUM({label}),2)))"
                f"SQRT(SUM(POWER(({col}-mean_{col}),2)) / COUNT({col}) )"
                for col in selected_columns
            ])
        query = f"""
            CREATE VIEW normalized_specified_typhoon_data AS (
            SELECT time, {columns}, {label}
            FROM specified_typhoon_data as t
            CROSS JOIN (
                SELECT
                    {', '.join([f"SUM({col}) / COUNT({col}) AS mean_{col}" for col in selected_columns])}
                FROM specified_typhoon_data
            ) AS minmax )
        """

        # params = (start_time, end_time)
        self.mycursor.execute(query)
        self.mydb.commit()

    def concat_data(self,concat_n,):
        combined_data = []
        combined_label = []

        embed_dim = len(self.raw_data[0][0][0])
        
        for datas,labels in self.raw_data: #each typhoon
            datas = torch.tensor(datas)
            for i,(x,y) in enumerate(zip(datas,labels)): #each data row
                data_ = torch.empty(concat_n,embed_dim)
                if concat_n-i-1>0:
                    data_[0:concat_n-i-1] = datas[0].repeat(concat_n-i-1,1)
                    data_[concat_n-i-1:] = datas[:i+1]
                else:
                    data_[:] = datas[i-concat_n+1:i+1]
                # label = y
                combined_data.append(data_)
                combined_label.append(torch.tensor(y))
        # print(len(combined_data),len(combined_label))
        self.x = torch.stack(combined_data)
        # self.y = torch.cat(combined_label)
        self.y = torch.stack(combined_label)
        # return self.x, self.y
    def get_all_data(self,):
        return self.x,self.y
    
    def correlation_coefficient(self,typhoon_ids,selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],label='TPB_level'):
        columns = ', '.join([
            f"(COUNT({col})*SUM({col}*{label}) - SUM({col})*SUM({label})) / SQRT((COUNT({col})*SUM(POWER({col}, 2))-POWER(SUM({col}),2))*(COUNT({label})*SUM(POWER({label}, 2))-POWER(SUM({label}),2)))"
            for col in selected_columns
        ])
        params = []
        condition = ""
        
        for typhoon_id in typhoon_ids:
            self.mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
            result = self.mycursor.fetchone()
            if result:
                start_time = result[0]
                end_time = result[1]
                features = ', '.join([f"{col}" for col in [*selected_columns,label]])
                cond = f"""
                    SELECT time, {features}
                    FROM training_data
                    WHERE (time >= %s AND time <= %s)
                """
                params.append(start_time)
                params.append(end_time)
                if condition == "":
                    condition += cond
                else:
                    condition += f"UNION ALL\n"
                    condition += cond

        # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化
        query = f"""
            SELECT {columns}
            FROM ({condition}) as t
        """

        # params = (start_time, end_time)
        self.mycursor.execute(query, params)
        result = self.mycursor.fetchall()

        for row in result:
            print(row)
            # print(type(row))
        return result[-1]
    
    def _correlation_coefficient(self,typhoon_ids,selected_columns = ['Shihimen', 'Feitsui', 'TPB', 'inflow', 'outflow', 'Feitsui_outflow', 'Tide'],label='TPB_level'):
        columns = ', '.join([
            f"(COUNT({col})*SUM({col}*{label}) - SUM({col})*SUM({label})) / SQRT((COUNT({col})*SUM(POWER({col}, 2))-POWER(SUM({col}),2))*(COUNT({label})*SUM(POWER({label}, 2))-POWER(SUM({label}),2)))"
            for col in selected_columns
        ])
        params = []
        condition = ""
        for typhoon_id in typhoon_ids:
            self.mycursor.execute("SELECT startTime, endTime FROM typhoon_list WHERE id = %s", (typhoon_id,))
            result = self.mycursor.fetchone()
            if result:
                start_time = result[0]
                end_time = result[1]
                cond = " (time >= %s AND time <= %s)"
                params.append(start_time)
                params.append(end_time)
                if condition == "":
                    condition += cond
                else:
                    condition += " OR"
                    condition += cond

        # 建立 SQL 查詢語句，從 training_data 資料表中選擇指定時間範圍內的資料並進行規範化  
        query = f"""
            SELECT {columns}
            FROM training_data
            WHERE {condition}
        """

        # params = (start_time, end_time)
        self.mycursor.execute(query, params)
        result = self.mycursor.fetchall()

        for row in result:
            print(row)
            # print(type(row))
        return result[-1]


if __name__ == '__main__':
    data = TyphoonDataset()
    data.create_db()
    data.load_db()
    data.fetch_data(['1','23','36'])
    data.concat_data(5)
    print(data.x.shape,data.y.shape)
    

        