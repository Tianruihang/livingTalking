###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import av
from fractions import Fraction

from redis_global import redis_manager
from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.result = None
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        
        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

    def put_msg_txt(self,msg,eventpoint=None):
        self.tts.put_msg_txt(msg,eventpoint)
    
    def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,eventpoint)

    def put_audio_file(self,filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        # 执行语音合成
        self.tts.flush_talk()
        # 执行语音识别
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        """通知事件点"""
        logger.info("notify:%s",eventpoint)
        #判断如何eventpoint中的status为end则删除缓存
        if eventpoint and eventpoint.get('status') == 'end':
            num = 1
            #获取当前帧
            mirror_index = redis_manager.get_current_frame("mirror_index")
            mirror_turn = redis_manager.get_current_frame("mirror_turn")
            # 变更视频需要改动的地方
            if mirror_index < 290:
                redis_manager.delete_max_cache_size(num)
            else:
                # 定义目标帧列表和阈值
                # 变更视频需要改动的地方
                target_frames = [290, 415, 540, 630, 730]
                threshold = 50  # 允许的帧偏差范围

                # 初始化最接近的目标帧
                chosen_target = None
                min_diff = float('inf')  # 初始最小偏差设为无穷大
                # 遍历所有目标帧，寻找最接近的合法目标
                for target in target_frames:
                    diff = abs(mirror_index - target)
                    # 检查是否在阈值范围内且偏差最小
                    if diff <= threshold and diff < min_diff:
                        min_diff = diff
                        chosen_target = target
                print('[INFO] chosen_target:', chosen_target)
                if mirror_turn > 0:
                    #正向循环
                    # 根据最接近的目标帧更新当前帧
                    if chosen_target is not None:
                        num = mirror_index - chosen_target
                        if num > 0 :
                            redis_manager.set_current_frame("current_frame", mirror_index - 1)
                            print(f'[INFO] notify end, mirror_index={mirror_index}, mirror_turn={mirror_turn}, num={num}')
                            num = num // 25
                        else:
                            redis_manager.set_current_frame("current_frame", mirror_index + 1)
                            num = 0 - num
                            print(f'[INFO] notify end, mirror_index={mirror_index}, mirror_turn={mirror_turn}, num={num}')
                            num = num // 25
                else:
                    #反向循环

                    # 根据最接近的目标帧更新当前帧
                    if chosen_target is not None:
                        num = chosen_target - mirror_index
                        if num > 0 :
                            redis_manager.set_current_frame("current_frame", mirror_index + 1)
                            print(f'[INFO] notify end, mirror_index={mirror_index}, mirror_turn={mirror_turn}, num={num}')
                            num = num // 25
                        else:
                            redis_manager.set_current_frame("current_frame", mirror_index - 1)
                            num = 0 - num
                            print(f'[INFO] notify end, mirror_index={mirror_index}, mirror_turn={mirror_turn}, num={num}')
                            num = num // 25
                print(f'[INFO] notify end, mirror_index={mirror_index}, mirror_turn={mirror_turn}, num={num}')
                redis_manager.delete_max_cache_size(num)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return
        print(f'[INFO] start recording video and audio, sessionid={self.opt.sessionid}')
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        print(f'[INFO]-------------------get_audio_stream {audiotype} index:{self.custom_audio_index[audiotype]}')
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
    def set_result_msg(self,msg):
        print('set_result_msg:',msg)
        self.result = msg
    def get_result_msg(self):
        msg = self.result
        self.result = None
        return msg
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1