# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response,llm_wenda_response,llm_java_wenda_response

import argparse
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger
from redis_global import redis_manager
from gpu_pool import GPUPool
from model_registry import Wav2LipRegistry

app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
# GPU pooling & model registry
session_gpu:Dict[int, int] = {}
gpu_pool:GPUPool = None
wav2lip_registry:Wav2LipRegistry = None
# Shared players per session to avoid multiple rendering threads
session_players:Dict[int, 'HumanPlayer'] = {}
# 记录每个 sessionid 对应的 RTCPeerConnection 数量，用于安全回收资源
session_pc_counts:Dict[int, int] = {}
# 跟踪每个 sessionid 关联的 RTCPeerConnection（便于主动断开推送）
session_pcs:Dict[int, set] = {}
opt = None
model = None
avatar = None
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        # Acquire GPU for the session if pool exists
        gi = gpu_pool.acquire(owner_id=f"session:{sessionid}") if gpu_pool is not None else -9999
        session_gpu[sessionid] = gi
        try:
            session_model = wav2lip_registry.get(gi) if (wav2lip_registry is not None and gi != -9999) else model
        except Exception:
            session_model = model
        nerfreal = LipReal(opt,session_model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 支持切换时复用已有 session，避免触发会话上限
    reuse_sessionid = params.get("sessionid")
    sessionid = None

    # accept string/int sessionid
    if reuse_sessionid is not None:
        try:
            reuse_sessionid = int(reuse_sessionid)
        except Exception:
            reuse_sessionid = None

    if reuse_sessionid is not None and reuse_sessionid in nerfreals:
        sessionid = reuse_sessionid
        logger.info('reuse sessionid=%d for handoff', sessionid)
    else:
        # 新建会话前检查上限
        if len(nerfreals) >= opt.max_session:
            # 当达到上限时，复用当前最空闲的会话（连接数最少）
            logger.info('reach max session, reuse least-loaded session')
            # 构建 (sid, load) 列表；没有计数的认为0
            if nerfreals:
                least_sid = min(nerfreals.keys(), key=lambda sid: session_pc_counts.get(sid, 0))
                sessionid = least_sid
                logger.info('reuse sessionid=%d (least-loaded)', sessionid)
            else:
                return web.Response(
                    status=429,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": "reach max session"})
                )
        else:
            sessionid = randN(6)
            logger.info('sessionid=%d', sessionid)
            nerfreals[sessionid] = None
            nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
            nerfreals[sessionid] = nerfreal
    
    pc = RTCPeerConnection()
    pcs.add(pc)

    # 增加该 session 的 RTCPeerConnection 引用计数
    session_pc_counts[sessionid] = session_pc_counts.get(sessionid, 0) + 1
    # 记录该 session 的 pc
    if sessionid not in session_pcs:
        session_pcs[sessionid] = set()
    session_pcs[sessionid].add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            # Clean up video track
            try:
                player.remove_video_track(video_track)
            except Exception:
                pass
            # 关闭时减少引用计数，计数为0再释放资源
            session_pc_counts[sessionid] = session_pc_counts.get(sessionid, 1) - 1
            if session_pc_counts[sessionid] <= 0:
                nerfreals.pop(sessionid, None)
                session_pc_counts.pop(sessionid, None)
                # Clean up shared player
                session_players.pop(sessionid, None)
                # release GPU if allocated
                gi = session_gpu.pop(sessionid, None)
                if gi is not None and gpu_pool is not None and gi != -9999:
                    try:
                        gpu_pool.release(gi)
                    except Exception:
                        pass
            # 从 session 映射中移除该 pc
            try:
                session_pcs.get(sessionid, set()).discard(pc)
                if not session_pcs.get(sessionid):
                    session_pcs.pop(sessionid, None)
            except Exception:
                pass
        if pc.connectionState == "closed":
            pcs.discard(pc)
            # Clean up video track
            try:
                player.remove_video_track(video_track)
            except Exception:
                pass
            session_pc_counts[sessionid] = session_pc_counts.get(sessionid, 1) - 1
            if session_pc_counts[sessionid] <= 0:
                nerfreals.pop(sessionid, None)
                session_pc_counts.pop(sessionid, None)
                # Clean up shared player
                session_players.pop(sessionid, None)
                gi = session_gpu.pop(sessionid, None)
                if gi is not None and gpu_pool is not None and gi != -9999:
                    try:
                        gpu_pool.release(gi)
                    except Exception:
                        pass
            # 从 session 映射中移除该 pc
            try:
                session_pcs.get(sessionid, set()).discard(pc)
                if not session_pcs.get(sessionid):
                    session_pcs.pop(sessionid, None)
            except Exception:
                pass

    # 检查 sessionid 是否存在
    if sessionid not in nerfreals:
        logger.error(f"Session {sessionid} not found in offer function")
        # 清理 RTCPeerConnection，避免资源占用
        try:
            await pc.close()
        except Exception:
            pass
        pcs.discard(pc)
        # Clean up video track if it was created
        try:
            if 'video_track' in locals():
                player.remove_video_track(video_track)
        except Exception:
            pass
        # 回收计数
        session_pc_counts[sessionid] = session_pc_counts.get(sessionid, 1) - 1
        if session_pc_counts.get(sessionid, 0) <= 0:
            session_pc_counts.pop(sessionid, None)
            nerfreals.pop(sessionid, None)
            # Clean up shared player
            session_players.pop(sessionid, None)
            gi = session_gpu.pop(sessionid, None)
            if gi is not None and gpu_pool is not None and gi != -9999:
                try:
                    gpu_pool.release(gi)
                except Exception:
                    pass
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Session creation failed"})
        )

    # Each session gets its own player to ensure independent video generation
    if sessionid not in session_players:
        session_players[sessionid] = HumanPlayer(nerfreals[sessionid])
    
    player = session_players[sessionid]
    audio_sender = pc.addTrack(player.audio)
    # Create a new video track for this connection to avoid frame sharing
    video_track = player.create_video_track()
    video_sender = pc.addTrack(video_track)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)
    logger.info(f'Codec preferences set: {[c.name for c in preferences]}')

    # 检测客户端是否为安卓设备并优化配置
    user_agent = request.headers.get('User-Agent', '')
    is_android = 'Android' in user_agent
    logger.info(f'Client User-Agent: {user_agent}, Android: {is_android}')

    # Limit and floor video bitrate/framerate to reduce oscillation and ramp-up time
    try:
        params = video_sender.getParameters()
        if not getattr(params, "encodings", None):
            # Ensure at least one encoding exists
            params.encodings = [{}]
        for enc in params.encodings:
            if is_android:
                # H264: 提高初始与上限码率，降低爬升期
                enc["maxBitrate"] = 5_000_000  # 5 Mbps cap for Android
                enc["maxFramerate"] = 24       # 24 fps for Android
                enc["scaleResolutionDownBy"] = 1.0  # 禁止分辨率缩放
                enc["minBitrate"] = 1_500_000  # 1.5 Mbps floor for Android
                logger.info('Applied Android-optimized video encoding parameters (H264 high start)')
            else:
                # 桌面设备提高初始与上限
                enc["maxBitrate"] = 9_000_000  # 9 Mbps cap
                enc["maxFramerate"] = 30       # allow 30 fps
                enc["scaleResolutionDownBy"] = 1.0  # 禁止分辨率缩放
                enc["minBitrate"] = 3_000_000  # 3 Mbps floor
        # aiortc.setParameters might be sync depending on version
        maybe = video_sender.setParameters(params)
        if hasattr(maybe, "__await__"):
            await maybe
    except Exception as e:
        logger.warning(f"setParameters for video_sender failed: {e}")

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # ---- SDP bandwidth hints injection (harmless for aiortc; helps receivers ramp up) ----
    def inject_sdp_bandwidth_hints(sdp: str) -> str:
        try:
            lines = sdp.split('\r\n')
            # map pt -> codec
            pt_to_codec = {}
            for line in lines:
                m = re.match(r"^a=rtpmap:(\d+)\\s+([A-Za-z0-9]+)", line)
                if m:
                    pt_to_codec[m.group(1)] = m.group(2).upper()

            # find video media section boundaries
            idx_m = None
            for i, line in enumerate(lines):
                if line.startswith('m=video '):
                    idx_m = i
                    break
            if idx_m is None:
                return sdp
            idx_end = len(lines)
            for i in range(idx_m+1, len(lines)):
                if lines[i].startswith('m='):
                    idx_end = i
                    break

            # ensure b=TIAS/AS in video section（为 H264 提高初始目标）
            has_b = False
            for i in range(idx_m+1, idx_end):
                if lines[i].startswith('b=TIAS:') or lines[i].startswith('b=AS:'):
                    # 根据设备类型设置不同的带宽（更高起步）
                    tias_bps = 5000000 if is_android else 9000000
                    as_kbps = tias_bps // 1000
                    # 覆盖 TIAS，并且紧随其后确保有 AS 行
                    lines[i] = f'b=TIAS:{tias_bps}'
                    # 检查下一行是否已有 AS，没有则插入
                    insert_as_at = i + 1
                    if insert_as_at < idx_end and not lines[insert_as_at].startswith('b=AS:'):
                        lines.insert(insert_as_at, f'b=AS:{as_kbps}')
                        idx_end += 1
                    has_b = True
                    break
            if not has_b:
                # insert after c= if exists, else right after m=
                insert_at = idx_m + 1
                for i in range(idx_m+1, idx_end):
                    if lines[i].startswith('c='):
                        insert_at = i + 1
                        break
                tias_bps = 5000000 if is_android else 9000000
                as_kbps = tias_bps // 1000
                lines.insert(insert_at, f'b=TIAS:{tias_bps}')
                lines.insert(insert_at + 1, f'b=AS:{as_kbps}')
                idx_end += 1

            # add x-google-* for VP8 payload types
            # find all fmtp lines and extend; if none, create
            vp8_pts = [pt for pt, codec in pt_to_codec.items() if codec == 'VP8']
            if vp8_pts:
                for pt in vp8_pts:
                    found = False
                    for i in range(idx_m+1, idx_end):
                        if lines[i].startswith(f'a=fmtp:{pt} '):
                            if 'x-google-start-bitrate' not in lines[i]:
                                # 根据设备类型设置不同的VP8参数
                                if is_android:
                                    lines[i] += ';x-google-start-bitrate=1000;x-google-min-bitrate=500;x-google-max-bitrate=2000'
                                else:
                                    lines[i] += ';x-google-start-bitrate=3000;x-google-min-bitrate=1500;x-google-max-bitrate=6000'
                            found = True
                            break
                    if not found:
                        # insert a new fmtp near rtpmap
                        insert_idx = idx_end
                        for i in range(idx_m+1, idx_end):
                            if lines[i].startswith(f'a=rtpmap:{pt} '):
                                insert_idx = i + 1
                        if is_android:
                            lines.insert(insert_idx, f'a=fmtp:{pt} x-google-start-bitrate=2000;x-google-min-bitrate=1500;x-google-max-bitrate=3000')
                        else:
                            lines.insert(insert_idx, f'a=fmtp:{pt} x-google-start-bitrate=5000;x-google-min-bitrate=3000;x-google-max-bitrate=8000')
                        idx_end += 1

            return '\r\n'.join(lines)
        except Exception as e:
            logger.warning(f"SDP injection failed: {e}")
            return sdp

    final_sdp = inject_sdp_bandwidth_hints(pc.localDescription.sdp)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
   
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": final_sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def gpu_status(request):
    # Get connection counts per session
    session_connections = {}
    for sid, pcs in session_pcs.items():
        session_connections[sid] = len(pcs)
    
    data = {
        "pool": gpu_pool.status() if gpu_pool is not None else {},
        "session_gpu": session_gpu,
        "active_sessions": list(nerfreals.keys()),
        "session_connections": session_connections,
        "session_players": {sid: len(player._HumanPlayer__video_tracks) for sid, player in session_players.items()},
    }
    return web.Response(content_type="application/json", text=json.dumps(data))

async def human(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        
        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )
        
        if params.get('interrupt'):
            nerfreals[sessionid].flush_talk()
        if params['type']=='echo':
            nerfreals[sessionid].put_msg_txt(params['text'])
        elif params['type']=='chat':
            #====================替换大模型==================
            await asyncio.get_event_loop().run_in_executor(None, llm_java_wenda_response, params['text'],nerfreals[sessionid])
            #nerfreals[sessionid].put_msg_txt(res)
            result_msg = nerfreals[sessionid].get_result_msg()
            # 检查是否为 None 或非字符串类型
            try:
                msg_dict = json.loads(result_msg)
                res = msg_dict.get("text", "")
                print("text内容是：", res)
            except json.JSONDecodeError:
                print("result_msg 不是合法的 JSON 格式")
            # 变更视频需要改动的地方
            redis_manager.set_max_cache_size(760)
            redis_manager.set_current_frame("current_frame",100)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data": res if 'res' in locals() else "ok"}
            ),
        )
    except Exception as e:
        logger.error(f"Error in human function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "Internal error", "data": str(e)}
            ),
        )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        
        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )
        
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.error(f"Error in humanaudio function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        
        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )
        
        nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data":"ok"}
            ),
        )
    except Exception as e:
        logger.error(f"Error in set_audiotype function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "Internal error", "data": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        
        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )
        
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data":"ok"}
            ),
        )
    except Exception as e:
        logger.error(f"Error in record function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "Internal error", "data": str(e)}
            ),
        )

async def is_speaking(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        
        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data": nerfreals[sessionid].is_speaking()}
            ),
        )
    except Exception as e:
        logger.error(f"Error in is_speaking function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "Internal error", "data": str(e)}
            ),
        )


async def stop_stream(request):
    try:
        params = await request.json()
        sessionid = params.get('sessionid', 0)

        # 未传 sessionid 或为 0/None 时，默认断开所有推送
        if not sessionid:
            close_tasks = []
            for pc in list(pcs):
                try:
                    close_tasks.append(pc.close())
                except Exception:
                    pass
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            pcs.clear()
            session_pcs.clear()
            session_pc_counts.clear()
            # release all GPUs and players
            if gpu_pool is not None:
                for sid, gi in list(session_gpu.items()):
                    try:
                        if gi != -9999:
                            gpu_pool.release(gi)
                    except Exception:
                        pass
                    session_gpu.pop(sid, None)
            # Clean up all shared players
            session_players.clear()
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "data": "stopped_all"}),
            )

        # 只断开指定 session 的推送
        if sessionid not in nerfreals and sessionid not in session_pcs:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": "Session not found", "data": "Session ID does not exist"}
                ),
            )

        close_tasks = []
        for pc in list(session_pcs.get(sessionid, set())):
            try:
                close_tasks.append(pc.close())
                pcs.discard(pc)
            except Exception:
                pass
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # 清理 session 的 pc 记录与计数，但保留 nerfreals 以便复用
        try:
            session_pcs.pop(sessionid, None)
        except Exception:
            pass
        if sessionid in session_pc_counts:
            session_pc_counts.pop(sessionid, None)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "data": "stopped"}),
        )
    except Exception as e:
        logger.error(f"Error in stop_stream function: {e}")
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "Internal error", "data": str(e)}),
        )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    session_pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    try:
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
        nerfreals[sessionid] = nerfreal

        pc = RTCPeerConnection()
        pcs.add(pc)
        # 记录该 session 的 pc
        if sessionid not in session_pcs:
            session_pcs[sessionid] = set()
        session_pcs[sessionid].add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)
                try:
                    session_pcs.get(sessionid, set()).discard(pc)
                    if not session_pcs.get(sessionid):
                        session_pcs.pop(sessionid, None)
                except Exception:
                    pass

        # 检查 sessionid 是否存在
        if sessionid not in nerfreals:
            logger.error(f"Session {sessionid} not found in run function")
            try:
                await pc.close()
            except Exception:
                pass
            pcs.discard(pc)
            return

        player = HumanPlayer(nerfreals[sessionid])
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())
        answer = await post(push_url,pc.localDescription.sdp)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
    except Exception as e:
        logger.error(f"Error in run function for session {sessionid}: {e}")
        # 清理资源
        if sessionid in nerfreals:
            nerfreals.pop(sessionid, None)



##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=4)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/stop_stream", stop_stream)
    appasync.router.add_get("/gpu/status", gpu_status)
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # init GPU pool & wav2lip registry
        global gpu_pool, wav2lip_registry
        gpu_pool = GPUPool()
        wav2lip_registry = Wav2LipRegistry("./models/wav2lip.pth")
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
