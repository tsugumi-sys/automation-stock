import requests
import json
import ccxt
import time

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'YOUR_API_KEY'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': notification_message }
    requests.post(line_notify_api, headers = headers, data = data)

def convertStrToJson(str):
    return json.loads(str)

def getPredVal(market, symbol, freq=7200):
    try:
        API_ENDPOINT = 'https://cryptowatch-lstm-api-d37w6addta-uc.a.run.app/api/predict?'
        url = API_ENDPOINT + f'market={market}&symbol={symbol}&freq={freq}' 
        response = requests.get(url)
        response = convertStrToJson(response.text)
        return response
    except:
        msg = '😡 Tried to get API of Croud Run, failed!!!'
        return None

def fetch_close_price(symbol, exchange):
    ticker = exchange.fetchTicker(symbol)
    price = ticker['last']
    return float(price)

def create_market_order(order_type, symbol, amount, exchange):
    order = exchange.createOrder(
        symbol=symbol,
        type='market',
        side=order_type,
        amount=amount,
    )
    return order

def calc_amount(price, amount=30):
    amount = amount / price
    if amount < 0.001:
        amount = round(amount, 5)
    elif amount < 0.01:
        amount = round(amount, 4)
    elif amount < 0.1:
        amount = round(amount, 3)
    elif amount < 1:
        amount = round(amount, 2)
    else:
        amount = round(amount)
    return amount

def main(data):
    try:
        # Initialize exchange with ccxt
        exchange_id = 'binance'
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET',
            'timeout': 30000,
            'enableRateLimit': True
        })
        
        
        # Pairs
        pairs = ['BTC/USDT', 'ETH/USDT', 'BCH/USDT', 'ETC/USDT', 'XMR/USDT', 'ADA/USDT', 'XLM/USDT', 'XRP/USDT', 'SOL/USDT', 'ZIL/USDT']

        # Check open order and cancell all of them
        for pair in pairs:
            open_orders = exchange.fetchOpenOrders(symbol = pair)
            if len(open_orders) > 0:
                for order in open_orders:
                    exchange.cancelOrder(symbol=pair, id=order['id'])


        balance = exchange.fetchBalance()['info']['balances']
        # check if long position and sell all of them
        for pair in pairs:
            asset = pair.replace('/USDT', '')
            price = fetch_close_price(pair, exchange)
            available = [i for i in balance if i['asset'] == asset]
            if len(available) > 0:
                available = float(available[0]['free'])
                print(asset, available)
            else:
                available = 0
            if price * available > 10:
                # market sell order
                create_market_order(order_type='sell', symbol=pair, amount=available, exchange=exchange)

        time.sleep(3)

        # Check Balance
        usdt_balance = exchange.fetchBalance()['info']['balances']

        # Available USDT
        usdt = [i for i in usdt_balance if i['asset'] == 'USDT']
        if len(usdt) > 0:
            usdt_available = usdt[0]['free']
        else:
            usdt_available = 0
        print(usdt_available)

        # Order Buy from the prediction value
        for pair in pairs:
            sym = pair.replace('/', '').lower()
            pred = getPredVal('binance', sym)
            if pred:
                pred = round(float(pred['predict']), 4)
                print(pred)
                send_line_notify(f'{pair}: {pred}')
                if pred > 0.7:
                    # Create Limit Order
                    try:
                        price = fetch_close_price(pair, exchange)
                        amount = calc_amount(price)
                        exchange.createOrder(
                            symbol=pair,
                            type='limit',
                            side='buy',
                            amount=amount,
                            price=price,
                        )
                        msg = f'🚀 Created Limit Order of {pair} at price {price}'
                        send_line_notify(msg)
                    except:
                        msg = f'😡 Tried to limit order of {pair}, but failed!!!'
                        send_line_notify(msg)
        send_line_notify('\n\n🎉 Binance BOT LONG \n\nSucccessfully Ended')
        return 'Finished'
    except:
        msg = f'😡 CCXT Initialization was failed!!!'
        send_line_notify(msg)
        return 'Finished with fail'

def get_val(data):
    try:
        # Pairs
        pairs = ['XMR/USDT', 'ADA/USDT', 'XLM/USDT', 'XRP/USDT', 'BCH/USDT', 'ETH/USDT', 'BTC/USDT', 'ETC/USDT']
        for pair in pairs:
            sym = pair.replace('/', '').lower()
            pred = getPredVal('binance', sym)
            if pred:
                pred = round(float(pred['predict']), 4)
                print(pred)
                send_line_notify(f'Long Model {pair}: {pred}')
        send_line_notify('\n\n🎉 Binance BOT LONG \n\nSucccessfully Ended')
        return 'Finished'
    except:
        msg = f'😡 CCXT Initialization was failed!!!'
        send_line_notify(msg)
        return 'Finished with fail'