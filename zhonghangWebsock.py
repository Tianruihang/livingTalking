# ✅ 必须放最前面
import eventlet
from anyio import sleep

from redis_global import redis_manager

eventlet.monkey_patch()
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return "WebSocket 服务运行中"

# WebSocket 连接事件
@socketio.on('connect')
def handle_connect():
    print('前端已连接')
    emit('message', {'text': '欢迎使用中行数字人！'})

# ✅ 新增：外部调用接口
@app.route('/api/push', methods=['POST'])
def api_push():
    data = request.get_json()
    text = data.get('text')
    sessionId = data.get('sessionId','')
    if not text:
        return jsonify({'success': False, 'error': '缺少 text 参数'}), 400

    # 通过 WebSocket 推送到所有连接的客户端
    socketio.emit('message', {'text': text,'sessionId':sessionId})
    return jsonify({'success': True, 'text': text})

REDIS_WAITING_KEYS="ceyan:questionWaiting:python"
# ✅ 可选：测试定时推送逻辑
def push_data():

    while True:
        feature_recorded = redis_manager.get_current_frame(REDIS_WAITING_KEYS)
        print(f'feature_recorded: {feature_recorded}')
        if feature_recorded != 0:
            #等待2s
            sleep(2)
            socketio.emit('message', {'text': f'开始可视化'})
        else:
            socketio.emit('message', {'text': f'停止可视化'})

        time.sleep(1)

if __name__ == '__main__':
    threading.Thread(target=push_data).start()  # 可选
    socketio.run(app, host='0.0.0.0', port=5010)
