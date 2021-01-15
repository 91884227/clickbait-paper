# Click-paper README
###### tags: `README`

# transfer_to_numeric.py
* 檔案在 `./CNN_Anomaly_model` 下
* 產生 embedding 後的資料放在 `./CNN_Anomaly_model/數值資料/<embedding>/ `
* usage
`transfer_to_numeric.py [--w w] [--t Bool]`
* parameter

| Column 1 | Column 2 | Column 3 |
| -------- | -------- | -------- |
| w    | 1 :  Glove <br> 2: w2V  | 2   |
| t    | True: Testing <br> False: not testing  | True    |


# ckip.py
* usage
```python=
from ckip import ckip
test = ckip()
print("去吃大便斷句結果", test("去吃大便"))
```

# read_embedding.py
* usage 
```python=
from read_embedding import glove_embedding, w2v_embedding
test1 = glove_embedding()
print("test1(\"韓國\"):", test1("韓國") )

test1 = w2v_embedding()
print("test1(\"韓國\"):", test1("韓國") )
```