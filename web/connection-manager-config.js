// WebRTC 连接管理器配置
window.ConnectionManagerConfig = {
    // 定期重启配置
    periodicRestart: {
        enabled: true,
        interval: 60000,        // 重启间隔 (1分钟)
        maxAttempts: 3,          // 最大重启次数
        minInterval: 60000       // 最小重启间隔 (1分钟)
    },

    // 智能切换配置
    smartSwitch: {
        enabled: false,
        // 基线比例阈值（保留兼容），但优先使用固定阈值
        bitrateThreshold: 0.5,   // 码率下降阈值 (50%)
        // 固定码率阈值与连续次数触发（优先使用）
        fixedBitrateKbps: 500,  // 固定码率阈值（kbps）
        lowCountToTrigger: 5,    // 连续低于阈值次数触发
        minSwitchInterval: 60000, // 最小切换间隔 (1分钟)
        gracePeriodMs: 30000,    // 切换宽限期：建立或刚切换后这段时间内不触发（避免爬升期）
        rttThreshold: 200,       // RTT阈值 (ms)
        packetLossThreshold: 0.1 // 丢包率阈值 (10%)
    },

    // 备用连接配置
    backupConnection: {
        enabled: true,
        preCreateDelay: 5000,   // 预创建延迟 (2秒)
        maxCreationTime: 30000,  // 最大创建时间 (30秒)
        retryAttempts: 2,        // 重试次数
        warmupMs: 2500,          // 切换前预热时长 (2.5秒)
        warmupMinBitrateKbps: 500, // 预热期希望达到的最低码率 (可选)
        // 控制主动预热切换，避免过早/频繁切换导致闪屏
        warmupSwitchEnabled: true,
        warmupSwitchCount: 1,
        warmupFirstDelayMs: 1200,
        warmupIntervalMs: 4000
    },
    
    // 连接质量评估
    qualityAssessment: {
        excellent: { rtt: 50, bitrate: 1000, packetLoss: 0.01 },
        good: { rtt: 100, bitrate: 500, packetLoss: 0.05 },
        fair: { rtt: 200, bitrate: 200, packetLoss: 0.1 },
        poor: { rtt: 200, bitrate: 200, packetLoss: 0.1 }
    },
    
    // 监控配置
    monitoring: {
        statsInterval: 3000,     // 统计监控间隔 (3秒)
        healthCheckInterval: 10000, // 健康检查间隔 (10秒)
        logLevel: 'info'         // 日志级别: debug, info, warn, error
    }
};

// 配置验证
ConnectionManagerConfig.validate = function() {
    var errors = [];
    
    if (this.periodicRestart.interval < this.periodicRestart.minInterval) {
        errors.push('重启间隔不能小于最小间隔');
    }
    
    if (this.smartSwitch.bitrateThreshold <= 0 || this.smartSwitch.bitrateThreshold >= 1) {
        errors.push('码率下降阈值必须在0-1之间');
    }
    
    if (this.smartSwitch.minSwitchInterval < 30000) {
        errors.push('最小切换间隔不能小于10秒');
    }
    
    if (errors.length > 0) {
        console.error('[配置验证失败]', errors);
        return false;
    }
    
    console.log('[配置验证成功]');
    return true;
};

// 动态配置更新
ConnectionManagerConfig.update = function(newConfig) {
    Object.assign(this, newConfig);
    if (this.validate()) {
        console.log('[配置更新成功]', newConfig);
        return true;
    }
    return false;
};

// 获取配置值
ConnectionManagerConfig.get = function(path) {
    return path.split('.').reduce((obj, key) => obj && obj[key], this);
};

// 导出配置
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConnectionManagerConfig;
}
