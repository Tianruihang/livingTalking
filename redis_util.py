import redis
import json
import pickle


class RedisManager:
    def __init__(self, host='localhost', port=6379, db=2):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.redis.config_set('maxmemory', '100mb')  # 设置为 100MB
        # 测试连接
        try:
            self.redis.ping()
            print("Redis connection established")
        except redis.ConnectionError:
            print("Redis connection failed")

    def set_state(self, session_id, key, value):
        """保存会话状态"""
        self.redis.hset(f"session:{session_id}", key, pickle.dumps(value))

    def get_state(self, session_id, key, default=None):
        """获取会话状态"""
        data = self.redis.hget(f"session:{session_id}", key)
        return pickle.loads(data) if data else default

    def cache_frame(self, session_id, frame_id, frame_data):
        """缓存视频帧"""
        self.redis.setex(f"frame:{session_id}:{frame_id}", 300, pickle.dumps(frame_data))

    def get_cached_frame(self, session_id, frame_id):
        """获取缓存的视频帧"""
        data = self.redis.get(f"frame:{session_id}:{frame_id}")
        return pickle.loads(data) if data else None

    def enqueue_task(self, queue_name, task_data):
        """添加任务到队列"""
        self.redis.rpush(queue_name, json.dumps(task_data))

    def dequeue_task(self, queue_name, timeout=0):
        """从队列获取任务"""
        _, data = self.redis.blpop(queue_name, timeout=timeout)
        return json.loads(data) if data else None
    #设置当前帧
    def set_current_frame(self, key, size):
        """设置当前帧"""
        self.redis.setex(key,60, size)
    #获取当前帧
    def get_current_frame(self,key):
        """获取当前帧"""
        try:
            if(self.redis.exists(key)):
                result = self.redis.get(key)
                return int(result)
            else:
                return 0
        except Exception as e:
            print(f"Error getting current frame: {str(e)}")
            return 0
    #删除当前帧
    def delete_current_frame(self,key):
        """删除当前帧"""
        self.redis.delete(key)

    #缓存最大数量
    def set_max_cache_size(self, size):
        """设置缓存最大数量"""
        self.redis.setex("max_cache_size",60, size)
   #删除最大数量
    def delete_max_cache_size(self,timeout=60):
        if timeout == 0 :
            #直接删除key
            self.redis.delete("max_cache_size")
        else:
            # 变更视频需要改动的地方
            self.redis.setex("max_cache_size",timeout, 760)
    #判断是否存在最大数量
    def has_max_cache_size(self):
        """判断是否存在最大数量"""
        try:
            if(self.redis.exists("max_cache_size")):
                result = self.redis.get("max_cache_size")
                return int(result)
            else:
                return 0
        except Exception as e:
            print(f"Error getting max cache size: {str(e)}")
            return 0
    def clear_session(self, session_id):
        """清除会话状态"""
        self.redis.delete(f"session:{session_id}")
        keys = self.redis.keys(f"frame:{session_id}:*")
        if keys:
            self.redis.delete(*keys)