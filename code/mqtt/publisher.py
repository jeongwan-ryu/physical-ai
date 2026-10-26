import json
import os
import random
import time
import paho.mqtt.client as mqtt

BROKER = os.getenv("BROKER", "broker")
TOPIC = os.getenv("TOPIC", "sensor/temperature")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="publisher")
client.connect(BROKER, 1883)
client.loop_start()  # 네트워크 처리를 백그라운드 스레드로

for i in range(1, 11):
    payload = json.dumps({"seq": i, "temp": round(random.uniform(20, 30), 1)})
    client.publish(TOPIC, payload, qos=1).wait_for_publish()
    print(f"[PUB] 발행 topic={TOPIC} payload={payload}")
    time.sleep(1)

client.loop_stop()
client.disconnect()
