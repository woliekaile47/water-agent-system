# Phase 2D-C-14：只读天气 API 沙箱

## 目的

本阶段只为未来真实部署补充 S6 天气数据源，不改变积水视觉识别、SAM2、岸线—DEM、水深反演或质量门。

项目当前不需要多个 API：

- 天气 API：用于真实部署时获取当前和未来 15/30/60 分钟降雨；
- 地图 API：当前不需要；
- 短信、邮件或通知 API：当前不需要；
- LLM API：不是积水测量和质量门的必要条件。

## 实现

使用 Open-Meteo Forecast API 的只读 HTTPS GET：

- `current=precipitation,rain,showers`
- `minutely_15=precipitation`
- `forecast_minutely_15=4`
- 单次超时，不循环重试；
- API 返回当前时间间隔内的降水量后，按响应中的 `interval` 换算为 mm/h；
- 未来 15/30/60 分钟降雨量由前 1/2/4 个十五分钟增量累加；
- 响应哈希、返回坐标、时区和数据来源写入 S6 JSON。

## 安全边界

- 默认的一键仿真链路继续读取仿真场景降雨，不读取实时天气；
- API 适配器不能发送通知；
- API 适配器不能生成真实预警；
- 请求失败时只有配置明确允许才使用回退值；
- 回退结果标记 `api_status=fallback`，不得伪装成实时数据；
- 示例经纬度只用于 API 连通性，不是实际部署位置；
- 用户确认部署位置前，不接入正式运行。

## 用户后续需要提供的信息

真实部署前只需提供：

1. 设备所在地的大致经纬度，精度到道路或校区即可；
2. 是否允许系统访问互联网；
3. 比赛或实际使用是否符合所选天气服务的许可条款。

本阶段不需要 API Key，不需要用户启动设备，也不需要开启真实预警。

## 联网验收

2026-07-26 从虚拟机执行一次只读 GET 请求，结果：

- provider：`open_meteo`
- API status：`success`
- 返回时区：`Asia/Shanghai`
- observation time：`2026-07-26T15:15`
- 当前降雨强度：0.0 mm/h
- 未来 15/30/60 分钟累计降雨：0.0 / 0.0 / 0.0 mm
- external notification allowed：`false`
- real warning allowed：`false`
- 输出仅写入 `/tmp/water_agent_c14_weather_sandbox`

数值只表示该 API 连通性演示坐标当时的天气，不代表项目真实部署地点，也不用于修改固定仿真场景。
