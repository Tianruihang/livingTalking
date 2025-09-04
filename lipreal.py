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

#from .utils import *
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp


from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal
from redis_global import redis_manager
#from imgcache import ImgCache

from tqdm import tqdm
from logger import logger

device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
print('=================================================Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path) #,weights_only=True
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	logger.info("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	model = model.to(device)
	return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    #self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    model_device = next(model.parameters()).device
    img_batch = torch.ones(batch_size, 6, modelres, modelres, device=model_device)
    mel_batch = torch.ones(batch_size, 1, 80, 16, device=model_device)
    model(mel_batch, img_batch)

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    redis_manager.set_current_frame("mirror_index", index)
    if turn % 2 == 0:
        redis_manager.set_current_frame("mirror_turn",1)
        return res
    else:
        redis_manager.set_current_frame("mirror_turn",-1)
        return size - res - 1 

def inference(quit_event,batch_size,face_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,model):
    
    #model = load_model("./models/wav2lip.pth")
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)
    
    #input_latent_list_cycle = torch.load(latents_out_path)
    length = len(face_list_cycle)
    index = 0
    count=0
    counttime=0
    logger.info('start inference')
    while not quit_event.is_set():
        starttime=time.perf_counter()
        mel_batch = []
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
            
        is_all_silence=True
        audio_frames = []
        #变更视频需要改动的地方
        length = 290
        if redis_manager.has_max_cache_size()  != 0:
            result = redis_manager.has_max_cache_size()
            #转int
            length = int(result)
            # print(f'redis_manager has_max_cache_size={length}')
        if redis_manager.get_current_frame("current_frame") != 0:
            index = redis_manager.get_current_frame("current_frame")
            # print(f'redis_manager get_current_frame={index}')
            redis_manager.delete_current_frame("current_frame")
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type==0:
                is_all_silence=False
        if is_all_silence:
            for i in range(batch_size):
                # print(f'__mirror_index(length,index) ={__mirror_index(length, index)}, length={length}, index={index}, i={i},batch_size : {batch_size}')
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # print('infer=======')
            t=time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                face = face_list_cycle[idx]
                img_batch.append(face)
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 只处理符合规则的数据，长度不一致则跳过本次
            if len(mel_batch) != len(img_batch) or len(mel_batch) != batch_size:
                logger.info(f"skip invalid batch: mel={len(mel_batch)}, img={len(img_batch)}, expected={batch_size}")
                continue

            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            
            model_device = next(model.parameters()).device
            img_batch = torch.as_tensor(np.transpose(img_batch, (0, 3, 1, 2)), dtype=torch.float32, device=model_device)
            mel_batch = torch.as_tensor(np.transpose(mel_batch, (0, 3, 1, 2)), dtype=torch.float32, device=model_device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            counttime += (time.perf_counter() - t)
            count += batch_size
            #_totalframe += 1
            if count>=100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count=0
                counttime=0
            for i,res_frame in enumerate(pred):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
            #print('total batch time:',time.perf_counter()-starttime)            
    logger.info('lipreal inference processor stop')

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        #self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        # self.W = opt.W
        # self.H = opt.H

        self.fps = opt.fps # 20 ms per frame
        
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size*2)  #mp.Queue
        #self.__loadavatar()
        self.model = model
        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = LipASR(opt,self)
        self.asr.warm_up()
        
        self.render_event = mp.Event()
        # Track if rendering is already started for this session
        self._rendering_started = False
        self._render_thread = None
    
    def __del__(self):
        logger.info(f'lipreal({self.sessionid}) delete')

   
    def process_frames(self,quit_event,loop=None,audio_track=None,player=None):
        
        while not quit_event.is_set():
            try:
                # print(f'infer silence res_frame_queue={self.res_frame_queue}')
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                    # if not self.custom_opt[audiotype].loop and self.custom_index[audiotype]>=len(self.custom_img_cycle[audiotype]):
                    #     self.curr_state = 1  #当前视频不循环播放，切换到静音状态
                else:
                    combine_frame = self.frame_list_cycle[idx]
                    #combine_frame = self.imagecache.get_img(idx)
            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                #combine_frame = copy.deepcopy(self.imagecache.get_img(idx))
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    continue
                #combine_frame = get_image(ori_frame,res_frame,bbox)
                #t=time.perf_counter()
                combine_frame[y1:y2, x1:x2] = res_frame
                #print('blending time:',time.perf_counter()-t)

            image = combine_frame #(outputs['image'] * 255).astype(np.uint8)
            image[0,:] &= 0xFE
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            
            # Broadcast frame to all active video tracks for this session only
            if player and hasattr(player, '_HumanPlayer__video_tracks'):
                video_tracks = list(player._HumanPlayer__video_tracks)
                if video_tracks:
                    logger.info(f"Session {self.sessionid}: Broadcasting frame to {len(video_tracks)} video tracks")
                    for video_track in video_tracks:
                        try:
                            # Check if track is still active before sending
                            if hasattr(video_track, '_queue') and not video_track._queue.full():
                                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
                            else:
                                logger.warning(f"Video track queue full or invalid, skipping frame")
                        except Exception as e:
                            logger.warning(f"Failed to send frame to video track: {e}")
                else:
                    logger.debug(f"No active video tracks to broadcast to for session {self.sessionid}")
            
            self.record_video_data(image)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate=16000
                # if audio_track._queue.qsize()>10:
                #     time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                self.record_audio_data(frame)
                #self.notify(eventpoint)
        logger.info('===============================lipreal process_frames thread stop================================')
            
    def render(self,quit_event,loop=None,audio_track=None,player=None):
        # Only start rendering once per LipReal instance
        if self._rendering_started:
            logger.info(f'Rendering already started for session {self.sessionid}, skipping')
            return
            
        self._rendering_started = True
        logger.info(f'Starting rendering for session {self.sessionid}')
        
        #if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,player))
        process_thread.start()
        #执行render方法
        print(f'========================lipreal render thread start, sessionid={self.sessionid},self.batch_size ={self.batch_size}========================')
        self._render_thread = Thread(target=inference, args=(quit_event,self.batch_size,self.face_list_cycle,
                                           self.asr.feat_queue,self.asr.output_queue,self.res_frame_queue,
                                           self.model,))
        self._render_thread.start()

        #self.render_event.set() #start infer process render
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set(): 
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # Check queue sizes for all video tracks to prevent overflow
            if player and hasattr(player, '_HumanPlayer__video_tracks'):
                max_queue_size = 0
                for video_track in list(player._HumanPlayer__video_tracks):
                    max_queue_size = max(max_queue_size, video_track._queue.qsize())
                
                if max_queue_size >= 5:
                    # logger.debug('sleep qsize=%d',max_queue_size)
                    time.sleep(0.04 * max_queue_size * 0.8)
                
            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        #self.render_event.clear() #end infer process render
        logger.info(f'lipreal thread stop for session {self.sessionid}')
            