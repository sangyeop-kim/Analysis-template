import boto3
import pandas as pd
from io import StringIO, BytesIO
import gzip
import getpass
import json
import os
from zipfile import ZipFile

class AWS_s3 :
    def __init__(self, region = 'ap-northeast-2', save_json = False) :
        
        try :
            with open("connect_info.json", "r") as json_file:
                key_info = json.load(json_file)
                aws_access_key_id = key_info['aws_access_key_id']
                aws_secret_access_key = key_info['aws_secret_access_key']

            self.bucket = self.access(aws_access_key_id, aws_secret_access_key, region)
            list(self.bucket.objects.all())
            print('접속 성공')

        except :
            while True :
                aws_access_key_id = getpass.getpass('Enter your aws_access_key_id:')
                aws_secret_access_key = getpass.getpass('Enter your aws_secret_access_key:')
                key_info = {'aws_access_key_id' : aws_access_key_id,
                           'aws_secret_access_key' : aws_secret_access_key}
                
                if save_json :
                    with open("connect_info.json", "w") as json_file:
                        json.dump(key_info, json_file)

                self.bucket = self.access(aws_access_key_id, aws_secret_access_key, region)
                try :
                    list(self.bucket.objects.all())
                    name = '접속 성공'
                    break

                except :
                    name = 'key가 잘못 입력됐습니다.'

                finally :
                    print(name)
        


    def access(self, aws_access_key_id, aws_secret_access_key, region) :
        session = boto3.Session(aws_access_key_id=aws_access_key_id, 
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region)
        s3 = session.resource('s3')
        return s3.Bucket('rtm-ai')
        
        
        
    def directory(self) :
        file_list = list(map(lambda x : x.key, self.bucket.objects.all()))

        unique_dir = []
        for i in [f.split('/')[:-1] for f in file_list] :
            if i not in unique_dir :
                unique_dir.append(i)

        name = ''
        new_dir = []

        for f in file_list :

            prefix = ('/').join(f.split('/')[:-1]) + '/'
            folder = (f.split('/')[-1]) == ''
            length = len([i for i in prefix.split('/') if len(i) > 0])

            if prefix.split('/')[:-2] in unique_dir and prefix not in new_dir:
                for i in range(len(prefix.split('/')[:-1])) :
                    if prefix.split('/')[:-1][:i+1] not in new_dir:
                        print()
                        print('    ' * (len(prefix.split('/')[:-1][:i+1]) - 2) + '├── ', end = '')
                        print(prefix.split('/')[:-1][:i+1][-1] + '/')
                        new_dir.append(prefix.split('/')[:-1][:i+1])
            else :
                print()
                
            if name != prefix and folder :
                name = prefix


                new_dir.append(name.split('/')[:-1])
                print('    ' * (length - 2), end = '')
                print(f)
            else :
                print('    ' * (length -1) + '├── ', end = '')
                print(f)
    
    def load_data(self, key, delete_zipfile = True) :
        '''
        example
        name : 'product-report/200310_product report_DB.csv'
        '''

        file_name = key.split('/')[-1]
        obj = self.bucket.Object(key = key)
        
        response = obj.get()

        lines = response['Body'].read()
        
        df_extension = ['csv']
        zip_extension = ['zip']
        ftr_extension = ['ftr']
        parquet_extension = ['parquet']
        

        if file_name.split('.')[-1] in df_extension :
            data = gzip.decompress(lines)
            df = pd.read_csv(StringIO(data.decode('utf-8')))

            return df

        elif file_name.split('.')[-1] in zip_extension :
            with open(file_name, 'wb') as zip_file :
                zip_file.write(lines)

            try :
                with ZipFile(file_name, 'r') as zipObj:
                    zipObj.extractall()
            except :
                raise Exception('압축을 풀지 못했습니다.')
            
            if delete_zipfile :
                os.remove(file_name)

            print('데이터를 불러왔습니다.')
        elif file_name.split('.')[-1] in ftr_extension:
            to_byte = BytesIO(lines)

            return pd.read_feather(to_byte)

        elif file_name.split('.')[-1] in parquet_extension:
            to_byte = BytesIO(lines)

            return pd.read_parquet(to_byte)
        