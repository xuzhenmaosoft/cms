import pandas as pd

from Train_model_DNN_Cluster import build_model_columns


def read_json():
  return  pd.read_json(path_or_buf=r'D:/0117_bak/functions-ai_0117.json',typ='series',orient='records')
def read_write():
  data = []
  with open(r'D:/0117_bak/functions-ai.json','r',encoding='utf-8') as f:
    while True:
      line = f.readline()
      #         print(line)
      if not line:
        break
      if line.find('/**') != -1 or line.find('//') != -1 or line.find('- m') != -1:
        pass
      else:
        data.append(line)
  with open(r'D:/0117_bak/functions-ai_0117.json','w') as w:
    for line in data:
      w.write(line)

  w.close()
  f.close()

if __name__ == '__main__':
  series = read_json()
  data = []
  for v in series:
    line = []
    # print('Algorithm :'+ v.get('name'))
    # print('returnType :'+v.get('returnType'))
    # get = v.get('paramTypes')
    # print('paramTypes :{get}'.format(get=get))
    name = v.get('name')
    line.append(name)
    type= v.get('returnType')
    line.append(type)
    para= v.get('paramTypes')
    line.append(para)
    data.append(line)
  df = pd.DataFrame(data,columns=['Algorithm','returnType','paramTypes'])
  df.to_csv(r'D:/0117_bak/result.csv',index=None)

