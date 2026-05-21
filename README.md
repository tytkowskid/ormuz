# Ormuz
Pamiętać by uzupełnić .env

Do odpalenia servera
```
python .\server\broker-server.py
```
W broker-server jest dict z przykładowymi cenami i początkową kasą jak coś to zmieniać tam

Przykład agenta
```
python main.py
```
Dane które bierze pod uwagę model są w TradeSignal i to jest TypedDict, ale w sumie nie musi na razie być więc jak coś zmieniasz to signal = zwykły słownik zadziała, a jak nie to zmienić w trade_signal.py
