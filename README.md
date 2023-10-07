# Split_Learning

## SL-FL
```bash
cd SL-FL_V3
nohup python -u server_communication.py > serverlog.log 2>&1 &
nohup python -u main.py -c ./utils/conf.json > log.log 2>&1 &
```
