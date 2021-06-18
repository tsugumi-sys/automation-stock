import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd

url = "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
res = requests.get(url, headers)
if res.status_code == 200:
    soup = bs(res.content, 'html.parser')
    tickers = []
    for item in soup.find_all('th', class_='vaM alC'):
        tic = item.get_text()
        if re.search('[A-Z]+', tic):
            tickers.append(tic)
        else:
            print(tic)
            continue
    df = pd.DataFrame({'symbol': tickers})
    df.to_csv('rakuten_us_stock.csv')
else:
    print('HTTP Response Code ', res.status_code)
    print(res.text)