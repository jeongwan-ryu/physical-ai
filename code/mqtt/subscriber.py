import os
import paho.mqtt.client as mqtt

BROKER = os.getenv("BROKER", "broker")
TOPIC = os.getenv("TOPIC", "sensor/#")  # '#' = 하위 토픽 전부


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[SUB] 브로커 연결: {reason_code}")
    client.subscribe(TOPIC, qos=1)  # 재연결 시에도 다시 구독되도록 on_connect 안에서
    print(f"[SUB] 구독 시작: {TOPIC}")


def on_message(client, userdata, msg):
    print(f"[SUB] 수신 topic={msg.topic} qos={msg.qos} retain={msg.retain} payload={msg.payload.decode()}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="subscriber")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_forever()
