/**
 * 安卓设备WebRTC兼容性修复脚本
 * 解决安卓浏览器无法正常显示视频的问题
 */

(function() {
    'use strict';
    
    // 检测是否为安卓设备
    function isAndroid() {
        return /Android/i.test(navigator.userAgent);
    }
    
    // 检测WebRTC支持
    function checkWebRTCSupport() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }
    
    // 检测Canvas支持
    function checkCanvasSupport() {
        var canvas = document.createElement('canvas');
        return !!(canvas.getContext && canvas.getContext('2d'));
    }
    
    // 修复视频元素属性
    function fixVideoElements() {
        var videos = document.querySelectorAll('video');
        videos.forEach(function(video) {
            // 设置必要的属性
            video.setAttribute('webkit-playsinline', 'true');
            video.setAttribute('x5-playsinline', 'true');
            video.setAttribute('x5-video-player-type', 'h5');
            video.setAttribute('x5-video-player-fullscreen', 'true');
            //video.muted = true;
            video.playsInline = true;
            video.autoplay = true;
            
            // 添加错误处理
            video.addEventListener('error', function(e) {
                console.error('[Android] Video error:', e);
            });
            
            video.addEventListener('loadstart', function() {
                console.log('[Android] Video loadstart');
            });
            
            video.addEventListener('loadedmetadata', function() {
                console.log('[Android] Video loadedmetadata');
            });
            
            video.addEventListener('canplay', function() {
                console.log('[Android] Video canplay');
            });
            
            video.addEventListener('playing', function() {
                console.log('[Android] Video playing');
            });
        });
    }
    
    // 修复Canvas渲染
    function fixCanvasRendering() {
        var canvas = document.getElementById('video-canvas');
        if (canvas) {
            // 启用硬件加速
            canvas.style.transform = 'translateZ(0)';
            canvas.style.backfaceVisibility = 'hidden';
            canvas.style.willChange = 'transform';
            
            // 设置Canvas上下文属性
            var ctx = canvas.getContext('2d', {
                alpha: false,
                desynchronized: false, // 安卓设备禁用
                willReadFrequently: true // 安卓设备启用
            });
            
            if (ctx) {
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
            }
        }
    }
    
    // 修复WebRTC配置
    function getAndroidWebRTCConfig() {
        return {
            sdpSemantics: 'unified-plan',
            iceTransportPolicy: 'all',
            bundlePolicy: 'max-bundle',
            rtcpMuxPolicy: 'require',
            iceServers: [
                { urls: ['stun:stun.l.google.com:19302'] }
            ]
        };
    }
    
    // 修复触摸事件
    function fixTouchEvents() {
        // 只防止双击缩放，不阻止滚动
        var lastTouchEnd = 0;
        document.addEventListener('touchend', function(e) {
            var now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                e.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
        
        // 移除阻止滚动的代码，允许正常滚动
        console.log('[Android] Touch events fixed - scrolling enabled');
    }
    
    // 修复页面视口
    function fixViewport() {
        var viewport = document.querySelector('meta[name="viewport"]');
        if (!viewport) {
            viewport = document.createElement('meta');
            viewport.name = 'viewport';
            document.head.appendChild(viewport);
        }
        viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
    }
    
    // 主修复函数
    function applyAndroidFixes() {
        if (!isAndroid()) {
            return;
        }
        
        console.log('[Android] Applying compatibility fixes...');
        
        // 检查支持情况
        var hasWebRTC = checkWebRTCSupport();
        var hasCanvas = checkCanvasSupport();
        
        console.log('[Android] WebRTC support:', hasWebRTC);
        console.log('[Android] Canvas support:', hasCanvas);
        
        if (!hasWebRTC) {
            console.error('[Android] WebRTC not supported!');
            return;
        }
        
        // 应用修复
        fixViewport();
        fixVideoElements();
        fixCanvasRendering();
        fixTouchEvents();
        
        console.log('[Android] Compatibility fixes applied');
    }
    
    // 导出函数供其他脚本使用
    window.AndroidCompat = {
        isAndroid: isAndroid,
        checkWebRTCSupport: checkWebRTCSupport,
        checkCanvasSupport: checkCanvasSupport,
        getAndroidWebRTCConfig: getAndroidWebRTCConfig,
        applyAndroidFixes: applyAndroidFixes
    };
    
    // 页面加载完成后自动应用修复
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyAndroidFixes);
    } else {
        applyAndroidFixes();
    }
    
})();
