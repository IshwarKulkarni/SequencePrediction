Running a simple Recurrent model on 
1. Stock <a href="https://iexcloud.io">data provided by IEX Cloud</a>. You will need the API Token from that website to be passed in with `--IEX_token=` along with the config file with `--config_file`.

2. Currency data provided by https://pkgstore.datahub.io 

How to run:

`python main.py  --config_file=./config_ticker.json --IEX_token=<>`

`python main.py  --config_file=./config_currency.json`


What to install:

`python -m pip install -r requirements.txt`

Go crazy.