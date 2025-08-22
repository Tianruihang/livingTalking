var pc = null;
var __webrtcStatsTimer = null;
var __lastVideoBytes = null;
var __lastVideoTs = null;
var __lastAudioBytes = null;
var __lastAudioTs = null;
var __canvasDrawer = null; // manages drawing and seamless switch
var pcBackup = null; // backup RTCPeerConnection for seamless restarts
var __periodicTimer = null;
var __restartInProgress = false;
var __cfg = (typeof window !== 'undefined' && window.ConnectionManagerConfig) ? window.ConnectionManagerConfig : {
    periodicRestart: { enabled: true, interval: 300000, maxAttempts: 3, minInterval: 60000 },
    backupConnection: { enabled: true, preCreateDelay: 10000, maxCreationTime: 30000, retryAttempts: 2 }
};
var __baselineVideoKbps = null;
var __lastSwitchAt = 0;
var __consecutiveLowCount = 0;
var __lastConnectAt = 0; // for grace period after start/switch
var __backupStats = null; // stats from backup pc during warmup

// 安卓兼容性检测
function __detectAndroidCompatibility() {
    var isAndroid = /Android/i.test(navigator.userAgent);
    var isMobile = /Mobile|Android|iPhone|iPad/i.test(navigator.userAgent);
    var hasWebRTC = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    var hasCanvas = !!document.createElement('canvas').getContext;
    
    console.log('[compat] Android:', isAndroid, 'Mobile:', isMobile, 'WebRTC:', hasWebRTC, 'Canvas:', hasCanvas);
    
    if (isAndroid && !hasWebRTC) {
        console.warn('[compat] Android device without WebRTC support');
        return false;
    }
    
    return true;
}

// 安卓优化配置
function __getAndroidOptimizedConfig() {
    var isAndroid = /Android/i.test(navigator.userAgent);
    if (!isAndroid) return {};
    
    return {
        // 降低 Canvas DPR 避免内存问题
        maxDPR: 1.0,
        // 禁用 RVFC（某些安卓不支持）
        disableRVFC: true,
        // 降低帧率限制
        maxFPS: 24,
        // 启用降级渲染
        enableFallback: true
    };
}

function __ts(){
    var d = new Date();
    var p = (n)=> (n<10?'0':'')+n;
    return d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
}

function __logSwitch(){
    try{
        var args = Array.prototype.slice.call(arguments);
        args.unshift('[switch '+__ts()+']');
        console.log.apply(console, args);
    }catch(e){ console.log('[switch]', arguments); }
}

function __initCanvasDrawer(){
    if (__canvasDrawer) return __canvasDrawer;
    var canvas = document.getElementById('video-canvas');
    var videoA = document.getElementById('video');
    var videoB = document.getElementById('video-buffer');
    if (!canvas || !videoA || !videoB) return null;

    var androidOpts = __getAndroidOptimizedConfig();
    var isAndroid = /Android/i.test(navigator.userAgent);
    
    // 安卓设备Canvas上下文配置优化
    var ctxOptions = { 
        alpha: false, 
        desynchronized: !isAndroid,  // 安卓设备禁用 desynchronized
        willReadFrequently: isAndroid  // 安卓设备启用 willReadFrequently
    };
    
    var ctx = canvas.getContext('2d', ctxOptions);
    if (ctx) {
        try { 
            ctx.imageSmoothingEnabled = true; 
            ctx.imageSmoothingQuality = 'high';
        } catch(e){}
    }
    
    // 安卓设备降低 DPR 避免内存问题
    var dpr = Math.min(window.devicePixelRatio || 1, androidOpts.maxDPR || 1.0);
    var current = videoA;
    var standby = videoB;
    var animationId = null;
    var rvfcSupported = !androidOpts.disableRVFC && typeof HTMLVideoElement !== 'undefined' && HTMLVideoElement.prototype && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
    var rvfcCancel = null; // keep track of the bound callback
    var lastDrawTs = 0;

    function resizeCanvas(){
        try{
            var container = document.querySelector('.video-container');
            if (!container) return;
            var rect = container.getBoundingClientRect();
            canvas.width = Math.round(rect.width * dpr);
            canvas.height = Math.round(rect.height * dpr);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            
            // 安卓设备特殊处理：强制重绘
            if (isAndroid) {
                setTimeout(function() {
                    try {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    } catch(e) {}
                }, 100);
            }
        }catch(e){}
    }

    function drawFrame(){
        if (!current || current.readyState < 2) {
            // 添加调试信息
            if (isAndroid) {
                console.log('[Android] Video not ready, state:', current ? current.readyState : 'no video');
            }
            return;
        }
        var vw = current.videoWidth || 0;
        var vh = current.videoHeight || 0;
        if (!vw || !vh) {
            if (isAndroid) {
                console.log('[Android] Video dimensions not available:', vw, 'x', vh);
            }
            return;
        }
        var cw = canvas.width;
        var ch = canvas.height;
        // cover fit
        var scale = Math.max(cw / vw, ch / vh);
        var dw = vw * scale;
        var dh = vh * scale;
        var dx = (cw - dw) / 2;
        var dy = (ch - dh) / 2;
        
        try {
            // 清除Canvas
            ctx.clearRect(0, 0, cw, ch);
            ctx.drawImage(current, dx, dy, dw, dh);
            
            // 安卓设备调试信息
            if (isAndroid && Math.random() < 0.01) { // 1%概率输出调试信息
                console.log('[Android] Canvas draw success:', vw, 'x', vh, '->', cw, 'x', ch);
            }
        } catch(e) {
            console.warn('Canvas drawImage failed:', e);
            // 安卓设备降级处理
            if (isAndroid) {
                setTimeout(drawFrame, 50);
            }
        }
    }

    function rafLoop(){
        animationId = requestAnimationFrame(rafLoop);
        // 安卓设备降低帧率以减少主线程压力
        var now = performance.now();
        var minInterval = isAndroid ? 50 : 33; // 安卓设备20fps，其他设备30fps
        if (now - lastDrawTs < minInterval) return;
        lastDrawTs = now;
        drawFrame();
    }

    function startRVFC(){
        if (!rvfcSupported || !current || isAndroid) return; // 安卓设备禁用RVFC
        // bind fresh callback on current element
        var cb = function(){
            drawFrame();
            // schedule next frame on the same current element
            try { current.requestVideoFrameCallback(cb); } catch(e){}
        };
        try { current.requestVideoFrameCallback(cb); } catch(e){}
        rvfcCancel = cb; // marker; no native cancel, but we overwrite on switch
    }

    function start(){
        if (animationId != null) cancelAnimationFrame(animationId);
        resizeCanvas();
        if (rvfcSupported && !isAndroid) {
            startRVFC();
        } else {
            rafLoop();
        }
        window.addEventListener('resize', resizeCanvas);
        document.addEventListener('fullscreenchange', resizeCanvas);
    }

    function stop(){
        if (animationId != null) cancelAnimationFrame(animationId);
        animationId = null;
        window.removeEventListener('resize', resizeCanvas);
        document.removeEventListener('fullscreenchange', resizeCanvas);
        try{ var c = canvas; c && c.getContext && c.getContext('2d').clearRect(0,0,c.width,c.height);}catch(e){}
    }

    function setStreamTo(videoEl, stream){
        if (!videoEl) return;
        try{
            if (videoEl.srcObject !== stream) {
                videoEl.srcObject = stream;
                // 确保视频元素能够正常播放（即使被隐藏）
                videoEl.style.display = 'block';
                videoEl.style.position = 'absolute';
                videoEl.style.top = '-9999px';
                videoEl.style.left = '-9999px';
                videoEl.style.width = '1px';
                videoEl.style.height = '1px';
                videoEl.style.opacity = '0';
                videoEl.style.visibility = 'hidden';
                videoEl.style.pointerEvents = 'none';
                videoEl.style.zIndex = '-1';
                
                // 安卓设备特殊处理
                if (isAndroid) {
                    videoEl.load(); // 强制重新加载
                    // 添加事件监听器来调试视频状态
                    videoEl.addEventListener('loadedmetadata', function() {
                        console.log('[Android] Video loadedmetadata:', videoEl.videoWidth, 'x', videoEl.videoHeight);
                    });
                    
                    videoEl.addEventListener('canplay', function() {
                        console.log('[Android] Video canplay');
                    });
                    
                    videoEl.addEventListener('playing', function() {
                        console.log('[Android] Video playing');
                    });
                    
                    videoEl.addEventListener('error', function(e) {
                        console.error('[Android] Video error:', e);
                    });
                    
                    videoEl.play().catch(function(e) {
                        console.warn('[Android] Video play failed:', e);
                    });
                } else {
                    // 非安卓设备也确保视频播放
                    videoEl.play().catch(function(e) {
                        console.warn('[Non-Android] Video play failed:', e);
                    });
                }
            }
        }catch(e){
            console.error('setStreamTo error:', e);
            try{ 
                videoEl.srcObject = null; 
                videoEl.srcObject = stream; 
                if (isAndroid) {
                    videoEl.load();
                    videoEl.play().catch(function(e) {
                        console.warn('[Android] Video play retry failed:', e);
                    });
                }
            }catch(_e){
                console.error('setStreamTo retry error:', _e);
            }
        }
    }

    function switchToStandby(){
        // swap references
        var tmp = current;
        current = standby;
        standby = tmp;
        // re-subscribe frame callback if using RVFC
        if (rvfcSupported && !isAndroid) {
            startRVFC();
        }
    }

    function updateStream(stream){
        // put new stream on standby video, wait for canplay, then swap
        setStreamTo(standby, stream);
        var done = false;
        var onReady = function(){
            if (done) return;
            done = true;
            standby.removeEventListener('canplay', onReady);
            switchToStandby();
        };
        if (standby.readyState >= 2) {
            switchToStandby();
        } else {
            standby.addEventListener('canplay', onReady, { once: true });
        }
    }

    function preloadStream(stream){
        // load to standby but DO NOT switch yet
        setStreamTo(standby, stream);
    }

    function commitSwitch(){
        return new Promise(function(resolve){
            if (standby.readyState >= 2) {
                switchToStandby();
                resolve(true);
                return;
            }
            var onReady = function(){
                standby.removeEventListener('canplay', onReady);
                switchToStandby();
                resolve(true);
            };
            standby.addEventListener('canplay', onReady, { once: true });
        });
    }

    __canvasDrawer = {
        start: start,
        stop: stop,
        updateStream: updateStream,
        preloadStream: preloadStream,
        commitSwitch: commitSwitch,
        resize: resizeCanvas
    };
    return __canvasDrawer;
}

function __ensureStatsPanel() {
    if (document.getElementById('webrtc-stats')) return;
    var panel = document.createElement('div');
    panel.id = 'webrtc-stats';
    panel.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:9999;background:rgba(0,0,0,0.6);color:#0f0;padding:8px 10px;border-radius:6px;font:12px/1.4 monospace;max-width:60vw;white-space:pre;pointer-events:none;will-change:transform;transform:translateZ(0);backface-visibility:hidden;contain:content;';
    panel.textContent = 'webrtc stats: waiting...';
    document.body.appendChild(panel);
}

function __updateStatsPanel(text) {
    var el = document.getElementById('webrtc-stats');
    if (!el) return;
    el.textContent = text;
}

function __startStatsMonitor() {
    if (!pc) return;
    window.__webrtcStats = null;
    __ensureStatsPanel();
    __lastVideoBytes = null;
    __lastVideoTs = null;
    __lastAudioBytes = null;
    __lastAudioTs = null;
    if (__webrtcStatsTimer) clearInterval(__webrtcStatsTimer);
    __webrtcStatsTimer = setInterval(async function () {
        if (!pc) return;
        try {
            var stats = await pc.getStats();
            var inboundVideo = null;
            var inboundAudio = null;
            var candidatePair = null;
            var trackVideo = null;
            stats.forEach(function (report) {
                if (report.type === 'inbound-rtp' && report.kind === 'video' && !report.isRemote) inboundVideo = report;
                if (report.type === 'inbound-rtp' && report.kind === 'audio' && !report.isRemote) inboundAudio = report;
                if (report.type === 'candidate-pair' && report.nominated && report.state === 'succeeded') candidatePair = report;
                if (report.type === 'track' && report.kind === 'video') trackVideo = report;
            });

            var nowMs = Date.now();
            var videoKbps = null, audioKbps = null, rttMs = null, availOutKbps = null;
            var width = trackVideo && trackVideo.frameWidth;
            var height = trackVideo && trackVideo.frameHeight;
            var fps = (inboundVideo && inboundVideo.framesPerSecond) || (trackVideo && trackVideo.framesPerSecond) || null;

            if (inboundVideo) {
                var vBytes = inboundVideo.bytesReceived || 0;
                if (__lastVideoBytes != null && __lastVideoTs != null) {
                    var deltaB = vBytes - __lastVideoBytes;
                    var deltaT = nowMs - __lastVideoTs;
                    if (deltaT > 0) videoKbps = Math.round((deltaB * 8) / deltaT);
                }
                __lastVideoBytes = vBytes;
                __lastVideoTs = nowMs;
            }

            if (inboundAudio) {
                var aBytes = inboundAudio.bytesReceived || 0;
                if (__lastAudioBytes != null && __lastAudioTs != null) {
                    var deltaAB = aBytes - __lastAudioBytes;
                    var deltaAT = nowMs - __lastAudioTs;
                    if (deltaAT > 0) audioKbps = Math.round((deltaAB * 8) / deltaAT);
                }
                __lastAudioBytes = aBytes;
                __lastAudioTs = nowMs;
            }

            if (candidatePair) {
                rttMs = candidatePair.currentRoundTripTime != null ? Math.round(candidatePair.currentRoundTripTime * 1000) : null;
                if (candidatePair.availableOutgoingBitrate != null) {
                    availOutKbps = Math.round(candidatePair.availableOutgoingBitrate / 1000);
                }
            }

            window.__webrtcStats = {
                videoKbps: videoKbps,
                audioKbps: audioKbps,
                availOutKbps: availOutKbps,
                rttMs: rttMs,
                width: width,
                height: height,
                fps: fps
            };

            var lines = [];
            lines.push('video: ' + (width || '?') + 'x' + (height || '?') + ' @ ' + (fps || '?') + 'fps');
            lines.push('  bitrate(v/a): ' + (videoKbps != null ? videoKbps : '?') + '/' + (audioKbps != null ? audioKbps : '?') + ' kbps');
            lines.push('  rtt: ' + (rttMs != null ? rttMs + ' ms' : '?') + '  availOut: ' + (availOutKbps != null ? availOutKbps + ' kbps' : '?'));
            __updateStatsPanel(lines.join('\n'));
            console.log('[webrtc-stats]', lines.join(' | '));

            // Smart switch: if bitrate drops below baseline*threshold for a while, trigger seamless restart
            try{
                var cfg = __cfg && __cfg.smartSwitch ? __cfg.smartSwitch : { enabled: true, bitrateThreshold: 0.5, fixedBitrateKbps: 1000, lowCountToTrigger: 3, minSwitchInterval: 30000, rttThreshold: 200 };
                var now = Date.now();
                if (typeof videoKbps === 'number' && videoKbps > 0){
                    if (__baselineVideoKbps == null){
                        __baselineVideoKbps = videoKbps;
                    } else {
                        // exponential moving baseline to adapt slowly
                        __baselineVideoKbps = Math.round(__baselineVideoKbps * 0.9 + videoKbps * 0.1);
                    }
                }
                var lowBitrate = false;
                if (cfg && cfg.enabled && typeof videoKbps === 'number'){
                    if (typeof cfg.fixedBitrateKbps === 'number'){
                        lowBitrate = videoKbps > 0 && videoKbps < cfg.fixedBitrateKbps;
                    } else if (typeof __baselineVideoKbps === 'number'){
                        var threshold = (__baselineVideoKbps || 1) * (cfg.bitrateThreshold || 0.5);
                        lowBitrate = videoKbps > 0 && videoKbps < threshold;
                    }
                }
                if (lowBitrate) { __consecutiveLowCount++; } else { __consecutiveLowCount = 0; }
                var longSinceLast = (now - __lastSwitchAt) > (cfg.minSwitchInterval || 30000);
                var graceOk = (now - (__lastConnectAt || 0)) > (cfg.gracePeriodMs || 0);
                var consecutiveTrigger = (__consecutiveLowCount >= (cfg.lowCountToTrigger || 3));
                if (cfg && cfg.enabled && longSinceLast && graceOk && (consecutiveTrigger || (rttMs != null && rttMs > (cfg.rttThreshold || 200)))){
                    var reasons = [];
                    if (consecutiveTrigger) reasons.push('low-bitrate');
                    if (rttMs != null && rttMs > (cfg.rttThreshold || 200)) reasons.push('high-rtt');
                    var sinceLastMs = now - __lastSwitchAt;
                    var sinceConnMs = now - (__lastConnectAt || 0);
                    __logSwitch(
                        'smart-switch trigger:', reasons.join('+') || 'unknown',
                        '| vKbps=', videoKbps,
                        '| baselineKbps=', __baselineVideoKbps,
                        '| fixedThresholdKbps=', (typeof cfg.fixedBitrateKbps==='number'?cfg.fixedBitrateKbps:'n/a'),
                        '| rttMs=', rttMs,
                        '| lowCount=', __consecutiveLowCount,
                        '| sinceLastSwitchMs=', sinceLastMs,
                        '| sinceConnectMs=', sinceConnMs,
                        '| graceOk=', graceOk,
                        '| minSwitchIntervalMs=', (cfg.minSwitchInterval||30000)
                    );
                    __lastSwitchAt = now;
                    __seamlessRestart('smart-switch');
                    __consecutiveLowCount = 0;
                }
            }catch(e){}
        } catch (e) {
            console.warn('getStats error:', e);
        }
    }, 3000);
}

function __stopStatsMonitor() {
    if (__webrtcStatsTimer) {
        clearInterval(__webrtcStatsTimer);
        __webrtcStatsTimer = null;
    }
}

function negotiateFor(targetPc, options){
    options = options || {};
    var updateSessionId = !!options.updateSessionId;
    var reuseSessionId = !!options.reuseSessionId;
    
    console.log('[WebRTC] Starting negotiation with options:', options);
    
    targetPc.addTransceiver('video', { direction: 'recvonly' });
    targetPc.addTransceiver('audio', { direction: 'recvonly' });
    
    console.log('[WebRTC] Transceivers added');
    
    return targetPc.createOffer().then(function(offer){
        console.log('[WebRTC] Offer created:', offer.type);
        return targetPc.setLocalDescription(offer);
    }).then(function(){
        console.log('[WebRTC] Local description set');
        return new Promise(function(resolve){
            if (targetPc.iceGatheringState === 'complete') {
                console.log('[WebRTC] ICE gathering already complete');
                return resolve();
            }
            var checkState = function(){
                console.log('[WebRTC] ICE gathering state:', targetPc.iceGatheringState);
                if (targetPc.iceGatheringState === 'complete'){
                    targetPc.removeEventListener('icegatheringstatechange', checkState);
                    console.log('[WebRTC] ICE gathering completed');
                    resolve();
                }
            };
            targetPc.addEventListener('icegatheringstatechange', checkState);
        });
    }).then(function(){
        var offer = targetPc.localDescription;
        var body = { sdp: offer.sdp, type: offer.type };
        if (reuseSessionId) {
            try{
                var sidStr = document.getElementById('sessionid') && document.getElementById('sessionid').value;
                var sid = parseInt(sidStr, 10);
                if (Number.isFinite(sid)) body.sessionid = sid;
            }catch(e){}
        }
        
        console.log('[WebRTC] Sending offer to server:', {
            type: body.type,
            sessionid: body.sessionid,
            sdpLength: body.sdp.length
        });
        
        return fetch('/offer', {
            body: JSON.stringify(body),
            headers: { 'Content-Type': 'application/json' },
            method: 'POST'
        });
    }).then(async function(response){
        console.log('[WebRTC] Server response status:', response.status);
        var text = await response.text();
        if (!response.ok){
            console.warn('Offer request failed:', response.status, text);
            throw new Error('Offer failed with status '+response.status+': '+text);
        }
        try{
            var json = JSON.parse(text);
            console.log('[WebRTC] Server response parsed:', {
                type: json.type,
                sessionid: json.sessionid,
                sdpLength: json.sdp ? json.sdp.length : 0
            });
            return json;
        } catch(parseErr){
            console.warn('Offer response is not valid JSON:', text);
            throw new Error('Invalid answer JSON: '+ parseErr.message);
        }
    })
      .then(function(answer){
        if (!answer || !answer.sdp || !answer.type){
            console.warn('Invalid answer structure:', answer);
            throw new Error('Invalid answer structure');
        }
        if (answer.type !== 'answer') answer.type = 'answer';
        try{ targetPc.__sessionid = answer.sessionid; }catch(e){}
        if (updateSessionId){
            try{ document.getElementById('sessionid').value = answer.sessionid; }catch(e){}
        }
        
        console.log('[WebRTC] Setting remote description');
        try{
            return targetPc.setRemoteDescription(answer);
        }catch(setErr){
            console.warn('setRemoteDescription failed with answer:', answer);
            throw setErr;
        }
    });
}

function __buildPeerConnection(config, onVideoTrack, attachAudio){
    var thePc = new RTCPeerConnection(config);
    var androidOpts = __getAndroidOptimizedConfig();
    var isAndroid = /Android/i.test(navigator.userAgent);
    
    thePc.addEventListener('track', function(evt){
        if (evt.track.kind === 'video'){
            if (typeof onVideoTrack === 'function') onVideoTrack(evt.streams[0]);
        } else if (attachAudio !== false) {
            try{ var a = document.getElementById('audio'); if (a) a.srcObject = evt.streams[0]; }catch(e){}
        }
    });
    
    // 安卓设备特殊错误处理和配置
    if (isAndroid) {
        thePc.addEventListener('connectionstatechange', function() {
            console.log('[Android] Connection state:', thePc.connectionState);
            if (thePc.connectionState === 'failed') {
                console.warn('[Android] Connection failed, trying fallback...');
                // 可以在这里添加降级逻辑
            }
        });
        
        thePc.addEventListener('iceconnectionstatechange', function() {
            console.log('[Android] ICE state:', thePc.iceConnectionState);
            if (thePc.iceConnectionState === 'failed') {
                console.warn('[Android] ICE failed, connection may be unstable');
            }
        });
        
        thePc.addEventListener('signalingstatechange', function() {
            console.log('[Android] Signaling state:', thePc.signalingState);
        });
        
        // 安卓设备特殊配置
        thePc.addEventListener('negotiationneeded', function() {
            console.log('[Android] Negotiation needed');
        });
    }
    
    return thePc;
}

function __schedulePeriodicRestart(){
    try{ if (!__cfg || !__cfg.periodicRestart || !__cfg.periodicRestart.enabled) return; }catch(e){ return; }
    if (__periodicTimer) clearTimeout(__periodicTimer);
    var interval = Math.max(__cfg.periodicRestart.minInterval || 60000, __cfg.periodicRestart.interval || 300000);
    __periodicTimer = setTimeout(function(){
        __seamlessRestart('periodic');
    }, interval);
}

function __seamlessRestart(reason){
    if (__restartInProgress) return;
    if (!__cfg || !__cfg.backupConnection || !__cfg.backupConnection.enabled) return;
    __restartInProgress = true;
    // create backup pc
    var config = { sdpSemantics: 'unified-plan' };
    try{ if (document.getElementById('use-stun').checked) config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }]; }catch(e){}
    __logSwitch('start seamless restart, reason =', reason);
    pcBackup = __buildPeerConnection(config, function(stream){
        var drawer = __initCanvasDrawer();
        if (drawer) drawer.preloadStream(stream);
    }, false);
    __logSwitch('backup PC created, negotiating offer...');
    negotiateFor(pcBackup, { updateSessionId: false, reuseSessionId: true }).then(function(){
        __logSwitch('backup PC remote description set, start warm-up...');
        // Collect backup stats during warm-up
        __backupStats = { videoKbps: null };
        var warmCfg = (__cfg && __cfg.backupConnection) ? __cfg.backupConnection : { warmupMs: 3000, warmupMinBitrateKbps: 1200 };
        var warmStart = Date.now();
        var warmTimer = setInterval(async function(){
            try{
                if (!pcBackup) return; // already swapped or failed
                var stats = await pcBackup.getStats();
                var inboundVideo = null;
                stats.forEach(function(r){ if (r.type==='inbound-rtp' && r.kind==='video' && !r.isRemote) inboundVideo = r; });
                if (inboundVideo){
                    if (!pcBackup.__lastVideoBytes || !pcBackup.__lastVideoTs){
                        pcBackup.__lastVideoBytes = inboundVideo.bytesReceived||0;
                        pcBackup.__lastVideoTs = Date.now();
                    } else {
                        var now = Date.now();
                        var deltaB = (inboundVideo.bytesReceived||0) - pcBackup.__lastVideoBytes;
                        var deltaT = now - pcBackup.__lastVideoTs;
                        if (deltaT>0) __backupStats.videoKbps = Math.round((deltaB*8)/deltaT);
                        pcBackup.__lastVideoBytes = inboundVideo.bytesReceived||0;
                        pcBackup.__lastVideoTs = now;
                    }
                }
            }catch(e){}
        }, 500);

        return new Promise(function(resolve){
            var done = false;
            var stop = function(){ if (done) return; done = true; try{ clearInterval(warmTimer); }catch(e){} resolve(true); };
            var check = function(){
                var elapsed = Date.now() - warmStart;
                var meetRate = (__backupStats && typeof __backupStats.videoKbps==='number' && __backupStats.videoKbps >= (warmCfg.warmupMinBitrateKbps||0));
                if (elapsed >= (warmCfg.warmupMs||3000) || meetRate){
                    __logSwitch('warm-up done. elapsed=', elapsed, 'ms, backup bitrate=', __backupStats.videoKbps);
                    stop();
                } else {
                    setTimeout(check, 300);
                }
            };
            setTimeout(check, 300);
        }).then(function(){
            __logSwitch('committing canvas switch after warm-up...');
            var drawer = __initCanvasDrawer();
            if (!drawer) throw new Error('Canvas drawer not ready');
            return drawer.commitSwitch();
        });
    }).then(function(){
        // swap active pc
        __logSwitch('canvas switched to backup stream, swapping PeerConnection');
        try{ document.getElementById('sessionid').value = pcBackup.__sessionid; }catch(e){}
        try{ if (pc) pc.close(); }catch(e){}
        pc = pcBackup;
        pcBackup = null;
        __stopStatsMonitor();
        __startStatsMonitor();
        __logSwitch('seamless restart done.');
        __lastConnectAt = Date.now();
    }).catch(function(e){
        console.warn('Seamless restart failed:', e);
        __logSwitch('seamless restart failed with error:', e && (e.stack || e.message || e));
        try{ if (pcBackup) pcBackup.close(); }catch(_e){}
        pcBackup = null;
    }).finally(function(){
        __restartInProgress = false;
        __schedulePeriodicRestart();
    });
}

function start() {
    // 检查兼容性
    if (!__detectAndroidCompatibility()) {
        alert('当前设备不支持 WebRTC，请使用支持 WebRTC 的浏览器');
        return;
    }

    var config = { sdpSemantics: 'unified-plan' };
    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }
    
    // 使用AndroidCompat模块的配置
    var isAndroid = /Android/i.test(navigator.userAgent);
    if (isAndroid && window.AndroidCompat) {
        config = window.AndroidCompat.getAndroidWebRTCConfig();
        console.log('[Android] Using AndroidCompat config:', config);
    } else {
        // 安卓设备优化配置
        var androidOpts = __getAndroidOptimizedConfig();
        if (androidOpts.maxFPS) {
            config.maxFramerate = androidOpts.maxFPS;
        }
    }
    
    pc = __buildPeerConnection(config, function(stream){
        console.log("Video track added");
        console.log("Stream info:", {
            id: stream.id,
            active: stream.active,
            tracks: stream.getTracks().map(t => ({
                kind: t.kind,
                enabled: t.enabled,
                readyState: t.readyState
            }))
        });
        
        // 检查调试模式
        var debugMode = document.getElementById('debug-mode') && document.getElementById('debug-mode').checked;
        
        // 安卓设备调试模式：暂时显示视频元素而不是Canvas
        if (isAndroid || debugMode) {
            console.log('[Android] Debug mode: showing video element instead of canvas');
            var video = document.getElementById('video');
            var canvas = document.getElementById('video-canvas');
            
            if (video && canvas) {
                // 隐藏Canvas，显示视频元素
                canvas.style.display = 'none';
                video.style.display = 'block';
                video.style.position = 'absolute';
                video.style.top = '0';
                video.style.left = '0';
                video.style.width = '100%';
                video.style.height = '100%';
                video.style.zIndex = '1';
                video.style.opacity = '1';
                video.style.visibility = 'visible';
                video.style.pointerEvents = 'auto';
                
                console.log('[Debug] Setting video stream to video element');
                video.srcObject = stream;
                video.load();
                video.play().catch(function(e) {
                    console.warn('[Android] Video play failed:', e);
                });
                
                // 添加视频状态监控
                video.addEventListener('loadedmetadata', function() {
                    console.log('[Android] Video loadedmetadata:', video.videoWidth, 'x', video.videoHeight);
                });
                
                video.addEventListener('canplay', function() {
                    console.log('[Android] Video canplay');
                });
                
                video.addEventListener('playing', function() {
                    console.log('[Android] Video playing');
                });
                
                video.addEventListener('error', function(e) {
                    console.error('[Android] Video error:', e);
                });
                
                return; // 跳过Canvas处理
            }
        }
        
        // 正常Canvas处理（非安卓设备或调试模式关闭）
        console.log('[Debug] Using Canvas mode');
        var drawer = __initCanvasDrawer();
        if (drawer){ 
            console.log('[Debug] Canvas drawer initialized');
            drawer.start(); 
            drawer.updateStream(stream); 
        } else {
            console.error('[Debug] Failed to initialize canvas drawer');
        }
        try{ 
            var video = document.getElementById('video');
            if (video) {
                console.log('[Debug] Setting video stream to hidden video element');
                video.srcObject = stream; 
                // 安卓设备特殊处理
                if (isAndroid) {
                    console.log('[Android] Setting video stream...');
                    video.load();
                    video.play().catch(function(e) {
                        console.warn('[Android] Video play failed:', e);
                    });
                    
                    // 检查视频流状态
                    setTimeout(function() {
                        console.log('[Android] Video state after 2s:', {
                            readyState: video.readyState,
                            videoWidth: video.videoWidth,
                            videoHeight: video.videoHeight,
                            paused: video.paused,
                            ended: video.ended
                        });
                    }, 2000);
                }
            } else {
                console.error('Video element not found');
            }
        }catch(e){
            console.error('Failed to set video srcObject:', e);
        }
    }, true);

    document.getElementById('start').style.display = 'none';
    __logSwitch('primary PC negotiating offer...');
    negotiateFor(pc, { updateSessionId: true, reuseSessionId: false }).then(function(){
        __logSwitch('primary PC remote description set');
        __lastConnectAt = Date.now();
        
        // 安卓设备特殊处理
        if (isAndroid) {
            console.log('[Android] WebRTC connection established');
            // 延迟检查视频状态
            setTimeout(function() {
                var video = document.getElementById('video');
                if (video && video.readyState >= 2) {
                    console.log('[Android] Video ready state:', video.readyState);
                } else {
                    console.warn('[Android] Video not ready, state:', video ? video.readyState : 'no video element');
                }
            }, 2000);
        }
    }).catch(function(e){
        console.warn('Primary negotiate failed:', e);
        __logSwitch('primary negotiate failed with error:', e && (e.stack || e.message || e));
        
        // 安卓设备特殊错误处理
        if (isAndroid) {
            console.error('[Android] WebRTC negotiation failed:', e);
            alert('安卓设备连接失败，请检查网络连接或刷新页面重试');
        } else {
            alert(e);
        }
    });
    document.getElementById('stop').style.display = 'inline-block';
    __startStatsMonitor();
    __schedulePeriodicRestart();
    
    // 安卓设备特殊处理
    if (isAndroid) {
        console.log('[Android] Fallback mode enabled');
    }
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
    __stopStatsMonitor();
    try{ var d = __initCanvasDrawer(); if (d) d.stop(); }catch(e){}
    if (__periodicTimer) { clearTimeout(__periodicTimer); __periodicTimer = null; }
    try{ if (pcBackup) pcBackup.close(); }catch(e){}
    pcBackup = null;
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
    __stopStatsMonitor();
    try{ if (pcBackup) pcBackup.close(); }catch(e){}
};

window.onbeforeunload = function (e) {
        setTimeout(() => {
                pc.close();
            }, 500);
        __stopStatsMonitor();
        e = e || window.event
        // 兼容IE8和Firefox 4之前的版本
        if (e) {
          e.returnValue = '关闭提示'
        }
        // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
        return '关闭提示'
      }